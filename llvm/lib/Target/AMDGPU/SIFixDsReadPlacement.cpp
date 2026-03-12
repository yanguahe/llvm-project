//===- SIFixDsReadPlacement.cpp - Hoist long-latency ds_reads -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FMHA kernels on gfx942 often issue LDS reads far ahead of their first use.
// Hoisting long-distance ds_read instructions earlier lets SIInsertWaitcnts
// regenerate weaker lgkmcnt values and reduce visible wait before later MFMA
// consumers.
//
// This pass is intentionally conservative:
//  * only DS loads are considered
//  * only instructions with a long first-use distance are moved
//  * movement is local to a basic block and stops at barriers / inline asm / CFG
//  * ReachingDefAnalysis must prove the motion is safe
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-ds-read-placement"

namespace {

class SIFixDsReadPlacement : public MachineFunctionPass {
public:
  static char ID;
  SIFixDsReadPlacement() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SI Fix DS Read Placement"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ReachingDefAnalysis>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setTracksLiveness();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char SIFixDsReadPlacement::ID = 0;

INITIALIZE_PASS_BEGIN(SIFixDsReadPlacement, DEBUG_TYPE, "SI Fix DS Read Placement",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(ReachingDefAnalysis)
INITIALIZE_PASS_END(SIFixDsReadPlacement, DEBUG_TYPE, "SI Fix DS Read Placement",
                    false, false)

FunctionPass *llvm::createSIFixDsReadPlacementPass() {
  return new SIFixDsReadPlacement();
}

static bool isHoistBoundary(const MachineInstr &MI) {
  if (MI.isInlineAsm() || MI.isCall() || MI.isPHI() || MI.isTerminator())
    return true;

  switch (MI.getOpcode()) {
  case AMDGPU::S_BARRIER:
  case AMDGPU::SCHED_BARRIER:
    return true;
  default:
    return false;
  }
}

static bool isHoistableDsRead(const MachineInstr &MI) {
  return SIInstrInfo::isDS(MI) && MI.mayLoad() && !MI.mayStore();
}

static SmallVector<Register, 4> getExplicitDefs(const MachineInstr &MI) {
  SmallVector<Register, 4> Defs;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg() || MO.isImplicit())
      continue;
    Defs.push_back(MO.getReg());
  }
  return Defs;
}

static unsigned getFirstUseDistance(const MachineInstr &MI,
                                    const SIRegisterInfo *TRI) {
  SmallVector<Register, 4> Defs = getExplicitDefs(MI);
  if (Defs.empty())
    return 0;

  unsigned Distance = 0;
  auto It = std::next(MI.getIterator());
  auto E = MI.getParent()->instr_end();
  for (; It != E; ++It) {
    ++Distance;
    for (Register Reg : Defs) {
      if (It->readsRegister(Reg, TRI))
        return Distance;
      if (It->modifiesRegister(Reg, TRI))
        return 0;
    }
  }
  return 0;
}

static bool tryHoist(MachineInstr &MI, ReachingDefAnalysis &RDA) {
  MachineBasicBlock &MBB = *MI.getParent();
  auto MIIt = MI.getIterator();
  if (MIIt == MBB.begin())
    return false;

  MachineInstr *BestTarget = nullptr;
  auto It = std::prev(MIIt);
  while (true) {
    MachineInstr &Cur = *It;
    if (!RDA.isSafeToMoveBackwards(&MI, &Cur))
      break;

    if (std::next(Cur.getIterator()) != MIIt)
      BestTarget = &Cur;

    if (It == MBB.begin() || isHoistBoundary(Cur))
      break;
    --It;
  }

  if (!BestTarget)
    return false;

  LLVM_DEBUG(dbgs() << "Hoist ds_read: " << MI << "  after: " << *BestTarget);
  MBB.splice(std::next(BestTarget->getIterator()), &MBB, MIIt);
  return true;
}

bool SIFixDsReadPlacement::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() >= AMDGPUSubtarget::GFX12)
    return false;

  const auto *TRI = ST.getRegisterInfo();
  auto &RDA = getAnalysis<ReachingDefAnalysis>();

  constexpr unsigned MinFirstUseDistance = 10;
  bool Modified = false;
  bool LocalChange = true;

  while (LocalChange) {
    LocalChange = false;

    for (MachineBasicBlock &MBB : MF) {
      SmallVector<MachineInstr *, 16> Candidates;
      for (MachineInstr &MI : MBB) {
        if (!isHoistableDsRead(MI))
          continue;
        unsigned Distance = getFirstUseDistance(MI, TRI);
        if (Distance < MinFirstUseDistance)
          continue;
        Candidates.push_back(&MI);
      }

      for (MachineInstr *MI : Candidates) {
        if (!MI->getParent())
          continue;
        if (!tryHoist(*MI, RDA))
          continue;

        Modified = true;
        LocalChange = true;
        RDA.reset();
        break;
      }

      if (LocalChange)
        break;
    }
  }

  return Modified;
}
