//===- SIFixDsReadInterleave.cpp - Split ds_read bursts into MFMA windows -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FMHA hot loops on gfx942 frequently materialize four consecutive ds_read
// instructions immediately before a chain of MFMA consumers. SIInsertWaitcnts
// then has to emit a full lgkmcnt staircase in front of that MFMA chain.
//
// This pass conservatively splits the burst by sinking the last ds_read to just
// before its first MFMA use when all of the following hold:
//   * the ds_read is the tail of a ds_read burst (next non-debug instruction is
//     an MFMA chain),
//   * its first use is a later MFMA in the same basic block,
//   * there are at least three earlier MFMAs before that first use, and
//   * the final MFMA->first-use window is otherwise empty.
//
// That rewrites the common shape
//   ds_read x4 ; mfma a ; mfma b ; mfma c ; mfma d(uses last read)
// into
//   ds_read x3 ; mfma a ; mfma b ; mfma c ; ds_read ; mfma d
// so SIInsertWaitcnts can often weaken the leading lgkmcnt sequence while
// preserving the MFMA consumer order.
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

#define DEBUG_TYPE "si-fix-ds-read-interleave"

namespace {

class SIFixDsReadInterleave : public MachineFunctionPass {
public:
  static char ID;
  SIFixDsReadInterleave() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SI Fix DS Read Interleave";
  }

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

char SIFixDsReadInterleave::ID = 0;

INITIALIZE_PASS_BEGIN(SIFixDsReadInterleave, DEBUG_TYPE,
                      "SI Fix DS Read Interleave", false, false)
INITIALIZE_PASS_DEPENDENCY(ReachingDefAnalysis)
INITIALIZE_PASS_END(SIFixDsReadInterleave, DEBUG_TYPE,
                    "SI Fix DS Read Interleave", false, false)

FunctionPass *llvm::createSIFixDsReadInterleavePass() {
  return new SIFixDsReadInterleave();
}

static bool isMotionBoundary(const MachineInstr &MI) {
  if (MI.isInlineAsm() || MI.isCall() || MI.isPHI() || MI.isTerminator())
    return true;

  if (SIInstrInfo::isWaitcnt(MI.getOpcode()))
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

static bool isMFMA(const MachineInstr &MI) { return SIInstrInfo::isMAI(MI); }

static SmallVector<Register, 4> getExplicitDefs(const MachineInstr &MI) {
  SmallVector<Register, 4> Defs;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg() || MO.isImplicit())
      continue;
    Defs.push_back(MO.getReg());
  }
  return Defs;
}

static MachineInstr *getNextNonDebugInstr(MachineInstr &MI) {
  auto It = std::next(MI.getIterator());
  auto E = MI.getParent()->instr_end();
  while (It != E && It->isDebugInstr())
    ++It;
  return It == E ? nullptr : &*It;
}

static MachineInstr *getFirstUse(MachineInstr &MI, const SIRegisterInfo *TRI) {
  SmallVector<Register, 4> Defs = getExplicitDefs(MI);
  if (Defs.empty())
    return nullptr;

  auto It = std::next(MI.getIterator());
  auto E = MI.getParent()->instr_end();
  for (; It != E; ++It) {
    if (It->isDebugInstr())
      continue;
    for (Register Reg : Defs) {
      if (It->readsRegister(Reg, TRI))
        return &*It;
      if (It->modifiesRegister(Reg, TRI))
        return nullptr;
    }
  }
  return nullptr;
}

static MachineInstr *getInterleaveTargetMFMA(MachineInstr &FirstUse,
                                             MachineInstr &Start,
                                             unsigned &EarlierMFMAs) {
  SmallVector<MachineInstr *, 8> MFMAs;
  auto It = std::next(Start.getIterator());
  while (It != FirstUse.getIterator()) {
    MachineInstr &Cur = *It;
    if (Cur.isDebugInstr()) {
      ++It;
      continue;
    }
    if (isMotionBoundary(Cur))
      return nullptr;
    if (isMFMA(Cur))
      MFMAs.push_back(&Cur);
    ++It;
  }

  EarlierMFMAs = MFMAs.size();
  if (EarlierMFMAs < 3)
    return nullptr;

  // Leave two MFMA instructions between the moved ds_read and its first use.
  return MFMAs[EarlierMFMAs - 3];
}

static bool hasOnlyMFMAsAndDebugBetween(const MachineInstr &From,
                                        const MachineInstr &To) {
  auto It = std::next(From.getIterator());
  while (It != To.getIterator()) {
    if (!It->isDebugInstr() && !isMFMA(*It))
      return false;
    ++It;
  }
  return true;
}

static bool tryInterleave(MachineInstr &MI, ReachingDefAnalysis &RDA,
                          const SIRegisterInfo *TRI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineInstr *Next = getNextNonDebugInstr(MI);
  if (!Next || !isMFMA(*Next))
    return false;

  MachineInstr *FirstUse = getFirstUse(MI, TRI);
  if (!FirstUse || !isMFMA(*FirstUse))
    return false;

  unsigned EarlierMFMAs = 0;
  MachineInstr *TargetMFMA = getInterleaveTargetMFMA(*FirstUse, MI, EarlierMFMAs);
  if (!TargetMFMA)
    return false;

  if (!hasOnlyMFMAsAndDebugBetween(*TargetMFMA, *FirstUse))
    return false;

  MachineInstr *InsertBefore = getNextNonDebugInstr(*TargetMFMA);
  if (!InsertBefore || !RDA.isSafeToMoveForwards(&MI, InsertBefore))
    return false;

  LLVM_DEBUG(dbgs() << "Interleave ds_read: " << MI
                    << "  after target MFMA: " << *TargetMFMA);

  auto MIIt = MI.getIterator();
  MBB.splice(InsertBefore->getIterator(), &MBB, MIIt);
  return true;
}

bool SIFixDsReadInterleave::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  if (ST.getGeneration() >= AMDGPUSubtarget::GFX12)
    return false;

  const auto *TRI = ST.getRegisterInfo();
  auto &RDA = getAnalysis<ReachingDefAnalysis>();

  bool Modified = false;
  bool LocalChange = true;

  while (LocalChange) {
    LocalChange = false;

    for (MachineBasicBlock &MBB : MF) {
      SmallVector<MachineInstr *, 16> Candidates;
      for (MachineInstr &MI : MBB) {
        if (!isHoistableDsRead(MI))
          continue;
        Candidates.push_back(&MI);
      }

      for (MachineInstr *MI : Candidates) {
        if (!MI->getParent())
          continue;
        if (!tryInterleave(*MI, RDA, TRI))
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
