//===- SIInsertWaveGroupPrio.cpp - Insert s_setprio for wave grouping -----===//
//
// Insert s_setprio around s_barrier instructions to implement wave group
// time-sharing on MI300X. Pattern matches the hand-written ASM kernel:
//   s_setprio 0   ; yield before barrier (let other group run)
//   s_barrier
//   ...           ; compute (GEMM, softmax, etc.)
//   s_setprio 1   ; reclaim priority after last MFMA
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-wave-group-prio"

static cl::opt<bool> EnableWaveGroupPrio(
    "amdgpu-wave-group-prio",
    cl::desc("Insert s_setprio for wave group time-sharing on MI300X"),
    cl::init(false), cl::Hidden);

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

bool SIInsertWaveGroupPrio::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableWaveGroupPrio)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX90AInsts())
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  // Detect loop blocks via back-edges.
  SmallPtrSet<MachineBasicBlock *, 16> LoopBlocks;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock *Succ : MBB.successors()) {
      if (Succ->getNumber() <= MBB.getNumber()) {
        // Back-edge found: mark all blocks from Succ to MBB as loop blocks.
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

    // Collect barriers and track last MFMA before each.
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

    // Also handle last MFMA before a back-edge branch (no trailing barrier).
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

    // Insert s_setprio around each barrier.
    for (auto &[Barrier, PrevMFMA] : BarrierMFMAPairs) {
      // s_setprio 1 after the last MFMA before this barrier.
      if (PrevMFMA) {
        auto InsertPt = std::next(MachineBasicBlock::iterator(PrevMFMA));
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_SETPRIO))
            .addImm(1);
        Changed = true;
      }

      // s_setprio 0 right before the barrier.
      BuildMI(MBB, *Barrier, DebugLoc(), TII->get(AMDGPU::S_SETPRIO))
          .addImm(0);
      Changed = true;
    }
  }

  return Changed;
}
