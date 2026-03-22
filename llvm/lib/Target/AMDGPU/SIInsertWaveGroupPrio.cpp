//===- SIInsertWaveGroupPrio.cpp - Insert s_setprio for wave grouping -----===//
//
// Insert s_setprio around s_barrier instructions to implement wave group
// time-sharing on MI300X. Pattern matches the hand-written ASM kernel:
//   s_setprio 0   ; yield before barrier (let other group run)
//   s_barrier
//   ...           ; compute (GEMM, softmax, etc.)
//   s_setprio 1   ; reclaim priority after last MFMA
//
// Also restores conditional branches around yield windows when the
// AMDGPU structurizer has converted them to predicated execution.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-wave-group-prio"

static cl::opt<bool> EnableWaveGroupPrio(
    "amdgpu-wave-group-prio",
    cl::desc("Insert s_setprio for wave group time-sharing on MI300X"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EnableYieldBranch(
    "amdgpu-yield-branch",
    cl::desc("Restore conditional branch around s_setprio 0 yield windows"),
    cl::init(true), cl::Hidden);

namespace {

class SIInsertWaveGroupPrio : public MachineFunctionPass {
public:
  static char ID;
  SIInsertWaveGroupPrio() : MachineFunctionPass(ID) {
    initializeSIInsertWaveGroupPrioPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "SI Insert Wave Group Priority";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool restoreYieldBranches(MachineFunction &MF, const SIInstrInfo *TII);
};

} // end anonymous namespace

char SIInsertWaveGroupPrio::ID = 0;

INITIALIZE_PASS(SIInsertWaveGroupPrio, DEBUG_TYPE,
                "SI Insert Wave Group Priority", false, false)

FunctionPass *llvm::createSIInsertWaveGroupPrioPass() {
  return new SIInsertWaveGroupPrio();
}

static bool isBarrierOpcode(unsigned Opc) {
  return Opc == AMDGPU::S_BARRIER ||
         Opc == AMDGPU::S_BARRIER_SIGNAL_M0 ||
         Opc == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_M0;
}

static bool isInlineAsmContaining(const MachineInstr &MI, StringRef Pattern) {
  if (!MI.isInlineAsm())
    return false;
  const MachineOperand &Op = MI.getOperand(InlineAsm::MIOp_AsmString);
  if (!Op.isSymbol())
    return false;
  return StringRef(Op.getSymbolName()).contains(Pattern);
}

/// Look for the pattern where StructurizeCFG/SimplifyCFG eliminated
/// the conditional branch around s_setprio 0 yield windows:
///
///   INLINEASM "s_setprio 0"      ; should be conditional on is_group_a
///   INLINEASM "s_nop 0xf"
///   INLINEASM "s_nop 0x7"
///   [SCHED_BARRIER ...]           ; optional scheduling fences
///   S_AND_B64 sD, sCond, EXEC    ; remnant of is_group_a condition test
///   S_CSELECT_B32 ...            ; uses SCC from above
///
/// Replace the 3 INLINEASMs with a single INLINEASM containing a
/// conditional branch (s_cbranch_scc0 3) that skips the yield window
/// for non-Group-A waves. The S_AND_B64 is moved before the INLINEASM
/// to set SCC for the branch condition.
///
bool SIInsertWaveGroupPrio::restoreYieldBranches(MachineFunction &MF,
                                                  const SIInstrInfo *TII) {
  bool Changed = false;

  struct YieldSite {
    MachineInstr *SetPrio;
    MachineInstr *Nop1;
    MachineInstr *Nop2;
    MachineInstr *CondAnd;
  };

  SmallVector<YieldSite, 8> Sites;

  for (MachineBasicBlock &MBB : MF) {
    for (auto MI = MBB.begin(), ME = MBB.end(); MI != ME; ++MI) {
      if (!isInlineAsmContaining(*MI, "s_setprio 0"))
        continue;

      auto It = std::next(MachineBasicBlock::iterator(MI));
      if (It == MBB.end() || !isInlineAsmContaining(*It, "s_nop 0xf"))
        continue;
      MachineInstr *Nop1 = &*It;

      It = std::next(It);
      if (It == MBB.end() || !isInlineAsmContaining(*It, "s_nop 0x7"))
        continue;
      MachineInstr *Nop2 = &*It;

      // Skip past SCHED_BARRIER, S_NOP, and meta instructions.
      It = std::next(It);
      while (It != MBB.end() &&
             (It->getOpcode() == AMDGPU::SCHED_BARRIER ||
              It->getOpcode() == AMDGPU::S_NOP ||
              It->isMetaInstruction()))
        ++It;

      if (It == MBB.end())
        continue;

      MachineInstr *CondInst = &*It;
      if (CondInst->getOpcode() != AMDGPU::S_AND_B64)
        continue;

      bool HasExecOp = false;
      for (unsigned i = 1; i < CondInst->getNumOperands(); ++i) {
        const MachineOperand &MO = CondInst->getOperand(i);
        if (MO.isReg() && MO.getReg() == AMDGPU::EXEC)
          HasExecOp = true;
      }
      if (!HasExecOp)
        continue;

      Sites.push_back({&*MI, Nop1, Nop2, CondInst});
    }
  }

  for (auto &Site : Sites) {
    MachineBasicBlock *MBB = Site.SetPrio->getParent();

    // Move the S_AND_B64 to before the yield INLINEASM so SCC is set
    // before the conditional branch executes.
    MBB->splice(MachineBasicBlock::iterator(Site.SetPrio),
                MBB, MachineBasicBlock::iterator(Site.CondAnd));

    // Replace the 3 INLINEASMs with one that embeds the conditional branch.
    // s_cbranch_scc0 3 skips 3 instructions (12 bytes = s_setprio + 2 NOPs).
    auto InsertPt = MachineBasicBlock::iterator(Site.SetPrio);
    auto DL = Site.SetPrio->getDebugLoc();
    BuildMI(*MBB, InsertPt, DL, TII->get(TargetOpcode::INLINEASM))
        .addExternalSymbol(
            "s_cbranch_scc0 3\n"
            "\ts_setprio 0\n"
            "\ts_nop 0xf\n"
            "\ts_nop 0x7")
        .addImm(InlineAsm::Extra_HasSideEffects);

    Site.SetPrio->eraseFromParent();
    Site.Nop1->eraseFromParent();
    Site.Nop2->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

bool SIInsertWaveGroupPrio::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX90AInsts())
    return false;

  if (!EnableWaveGroupPrio && !EnableYieldBranch)
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  if (EnableYieldBranch)
    Changed |= restoreYieldBranches(MF, TII);

  if (!EnableWaveGroupPrio)
    return Changed;

  MF.RenumberBlocks();

  SmallPtrSet<MachineBasicBlock *, 16> LoopBlocks;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock *Succ : MBB.successors()) {
      if (Succ->getNumber() <= MBB.getNumber()) {
        for (auto It = MachineFunction::iterator(Succ),
                  E = std::next(MachineFunction::iterator(&MBB));
             It != E; ++It) {
          LoopBlocks.insert(&*It);
        }
      }
    }
  }

  for (MachineBasicBlock &MBB : MF) {
    if (!LoopBlocks.count(&MBB))
      continue;

    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> BarrierMFMAPairs;
    MachineInstr *LastMFMA = nullptr;

    for (MachineInstr &MI : MBB) {
      if (SIInstrInfo::isMFMA(MI))
        LastMFMA = &MI;

      if (isBarrierOpcode(MI.getOpcode())) {
        BarrierMFMAPairs.push_back({&MI, LastMFMA});
        LastMFMA = nullptr;
      }
    }

    if (LastMFMA) {
      bool HasBackEdge = false;
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ->getNumber() <= MBB.getNumber()) {
          HasBackEdge = true;
          break;
        }
      }
      if (HasBackEdge) {
        auto InsertPt = std::next(MachineBasicBlock::iterator(LastMFMA));
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_SETPRIO))
            .addImm(1);
        Changed = true;
      }
    }

    for (auto &[Barrier, PrevMFMA] : BarrierMFMAPairs) {
      if (PrevMFMA) {
        auto InsertPt = std::next(MachineBasicBlock::iterator(PrevMFMA));
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_SETPRIO))
            .addImm(1);
        Changed = true;
      }

      BuildMI(MBB, *Barrier, DebugLoc(), TII->get(AMDGPU::S_SETPRIO))
          .addImm(0);
      Changed = true;
    }
  }

  return Changed;
}
