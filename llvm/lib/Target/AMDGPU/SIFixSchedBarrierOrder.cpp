//===- SIFixSchedBarrierOrder.cpp - Consolidate MFMAs before buffer_loads -===//
//
// Post-hazard-recognizer pass that separates V-prefetch buffer_loads from
// GEMM1 MFMA chains by hoisting MFMAs over v_add_u32+buffer_load blocks.
//
// Currently a no-op in practice: the actual compilation already places
// buffer_loads correctly in Phase 5 (after MFMA chain). The ISA dump showing
// interleaving is from a separate LLVM compilation with different RA.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sched-barrier-order"

namespace {

class SIFixSchedBarrierOrder : public MachineFunctionPass {
public:
  static char ID;
  SIFixSchedBarrierOrder() : MachineFunctionPass(ID) {
    initializeSIFixSchedBarrierOrderPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "SI Fix SCHED_BARRIER Ordering";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const SIRegisterInfo *TRI = nullptr;

  bool processBlock(MachineBasicBlock &MBB, const SIInstrInfo &TII);

  static bool isSBarrier(const MachineInstr &MI) {
    return MI.getOpcode() == AMDGPU::S_BARRIER;
  }

  static bool isGlobalBufferLoad(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::BUFFER_LOAD_DWORD_OFFEN ||
           Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN ||
           Opc == AMDGPU::BUFFER_LOAD_DWORDX4_OFFEN ||
           Opc == AMDGPU::BUFFER_LOAD_DWORD_OFFSET ||
           Opc == AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET ||
           Opc == AMDGPU::BUFFER_LOAD_DWORDX4_OFFSET;
  }

  static bool isMFMA(const MachineInstr &MI) {
    return SIInstrInfo::isMAI(MI);
  }

  static bool isVALU_AddU32(const MachineInstr &MI) {
    return MI.getOpcode() == AMDGPU::V_ADD_U32_e32 ||
           MI.getOpcode() == AMDGPU::V_ADD_U32_e64;
  }

  static bool isNopOrWaitcnt(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::S_NOP ||
           Opc == AMDGPU::S_WAITCNT ||
           Opc == AMDGPU::S_WAITCNT_VSCNT ||
           Opc == AMDGPU::SCHED_BARRIER;
  }

  static bool isAddrOrLoad(const MachineInstr &MI) {
    return isVALU_AddU32(MI) || isGlobalBufferLoad(MI);
  }

  static bool isDSPermute(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::DS_PERMUTE_B32 || Opc == AMDGPU::DS_BPERMUTE_B32;
  }

  static bool isSimpleCrossableInstr(const MachineInstr &MI) {
    return !MI.isMetaInstruction() && !MI.isTerminator() && !MI.isBranch() &&
           !MI.isCall() && !MI.isInlineAsm() && !MI.mayLoadOrStore() &&
           !isSBarrier(MI) && !isDSPermute(MI) && !isMFMA(MI);
  }

  bool regsConflict(const MachineInstr &A, const MachineInstr &B) const {
    for (const MachineOperand &Def : A.operands()) {
      if (!Def.isReg() || !Def.isDef())
        continue;
      for (const MachineOperand &Use : B.operands()) {
        if (!Use.isReg() || !Use.isUse())
          continue;
        if (TRI->regsOverlap(Use.getReg(), Def.getReg()))
          return true;
      }
    }
    for (const MachineOperand &Def : B.operands()) {
      if (!Def.isReg() || !Def.isDef())
        continue;
      for (const MachineOperand &Use : A.operands()) {
        if (!Use.isReg() || !Use.isUse())
          continue;
        if (TRI->regsOverlap(Use.getReg(), Def.getReg()))
          return true;
      }
    }
    for (const MachineOperand &DefA : A.operands()) {
      if (!DefA.isReg() || !DefA.isDef())
        continue;
      for (const MachineOperand &DefB : B.operands()) {
        if (!DefB.isReg() || !DefB.isDef())
          continue;
        if (TRI->regsOverlap(DefA.getReg(), DefB.getReg()))
          return true;
      }
    }
    return false;
  }

  bool hoistMFMAOverDSPermute(MachineBasicBlock &MBB);

  bool readsAnyDefOf(const MachineInstr &MI,
                     ArrayRef<MachineInstr *> Defs) const {
    for (const MachineInstr *D : Defs) {
      for (const MachineOperand &Def : D->operands()) {
        if (!Def.isReg() || !Def.isDef())
          continue;
        for (const MachineOperand &Use : MI.operands()) {
          if (!Use.isReg() || !Use.isUse())
            continue;
          if (TRI->regsOverlap(Use.getReg(), Def.getReg()))
            return true;
        }
      }
    }
    return false;
  }
};

} // end anonymous namespace

char SIFixSchedBarrierOrder::ID = 0;

INITIALIZE_PASS(SIFixSchedBarrierOrder, DEBUG_TYPE,
                "SI Fix SCHED_BARRIER Ordering", false, false)

FunctionPass *llvm::createSIFixSchedBarrierOrderPass() {
  return new SIFixSchedBarrierOrder();
}

bool SIFixSchedBarrierOrder::processBlock(MachineBasicBlock &MBB,
                                          const SIInstrInfo &TII) {
  bool Changed = false;

  SmallVector<MachineInstr *, 16> HWBarriers;
  for (MachineInstr &MI : MBB) {
    if (isSBarrier(MI))
      HWBarriers.push_back(&MI);
  }

  if (HWBarriers.size() < 2)
    return false;

  for (unsigned Ri = 0; Ri + 1 < HWBarriers.size(); ++Ri) {
    MachineInstr *FirstBarrier = HWBarriers[Ri];
    MachineInstr *NextBarrier = HWBarriers[Ri + 1];

    SmallVector<MachineInstr *, 8> BufLoads;
    for (auto It = std::next(FirstBarrier->getIterator());
         It != NextBarrier->getIterator(); ++It) {
      if (isGlobalBufferLoad(*It)) {
        bool WritesVGPR = false;
        for (const MachineOperand &MO : It->operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
            if (TRI->isVGPR(MBB.getParent()->getRegInfo(), MO.getReg()))
              WritesVGPR = true;
          }
        }
        if (WritesVGPR)
          BufLoads.push_back(&*It);
      }
    }

    if (BufLoads.empty())
      continue;

    bool HasMFMAAfterBufLoad = false;
    MachineInstr *LastBufLoad = BufLoads.back();
    for (auto It = std::next(LastBufLoad->getIterator());
         It != NextBarrier->getIterator(); ++It) {
      if (isMFMA(*It)) {
        HasMFMAAfterBufLoad = true;
        break;
      }
    }

    if (!HasMFMAAfterBufLoad)
      continue;

    MachineInstr *FirstBufLoad = BufLoads.front();
    auto ZoneStart = FirstBufLoad->getIterator();
    while (ZoneStart != std::next(FirstBarrier->getIterator())) {
      auto Prev = std::prev(ZoneStart);
      if (isAddrOrLoad(*Prev) || isNopOrWaitcnt(*Prev)) {
        ZoneStart = Prev;
      } else if (isMFMA(*Prev)) {
        auto CheckIt = Prev;
        bool HasAddrBefore = false;
        while (CheckIt != std::next(FirstBarrier->getIterator())) {
          auto Prev2 = std::prev(CheckIt);
          if (isAddrOrLoad(*Prev2)) {
            HasAddrBefore = true;
            break;
          } else if (isNopOrWaitcnt(*Prev2) || isMFMA(*Prev2)) {
            CheckIt = Prev2;
          } else {
            break;
          }
        }
        if (HasAddrBefore) {
          ZoneStart = Prev;
        } else {
          break;
        }
      } else {
        break;
      }
    }

    auto ZoneEnd = LastBufLoad->getIterator();
    while (ZoneEnd != std::prev(NextBarrier->getIterator())) {
      auto Next = std::next(ZoneEnd);
      if (Next == NextBarrier->getIterator())
        break;
      if (isMFMA(*Next) || isNopOrWaitcnt(*Next)) {
        ZoneEnd = Next;
      } else {
        break;
      }
    }

    SmallVector<MachineInstr *, 8> ZoneMFMAs;
    SmallVector<MachineInstr *, 16> ZoneAddrLoads;
    for (auto It = ZoneStart; It != std::next(ZoneEnd); ++It) {
      if (isMFMA(*It))
        ZoneMFMAs.push_back(&*It);
      else if (isAddrOrLoad(*It))
        ZoneAddrLoads.push_back(&*It);
    }

    if (ZoneMFMAs.empty() || ZoneAddrLoads.empty())
      continue;

    bool AllSafe = true;
    for (MachineInstr *MI : ZoneMFMAs) {
      if (readsAnyDefOf(*MI, ZoneAddrLoads)) {
        AllSafe = false;
        break;
      }
    }
    if (!AllSafe)
      continue;

    auto InsertPt = ZoneStart;
    for (MachineInstr *MI : ZoneMFMAs) {
      MBB.splice(InsertPt, &MBB, MI);
      Changed = true;
    }
  }

  return Changed;
}

bool SIFixSchedBarrierOrder::hoistMFMAOverDSPermute(MachineBasicBlock &MBB) {
  bool Changed = false;
  const MachineFunction *MF = MBB.getParent();

  for (auto It = MBB.begin(); It != MBB.end(); ++It) {
    unsigned Opc = It->getOpcode();
    if (Opc == AMDGPU::DS_PERMUTE_B32 || Opc == AMDGPU::DS_BPERMUTE_B32) {
      MachineInstr &Perm = *It;
      auto Next = std::next(It);
      while (Next != MBB.end() && isNopOrWaitcnt(*Next))
        ++Next;
      if (Next == MBB.end()) {
        continue;
      }
      if (!isMFMA(*Next)) {
        auto Mid = Next;
        auto MFMAIt = std::next(Mid);
        while (MFMAIt != MBB.end() && isNopOrWaitcnt(*MFMAIt))
          ++MFMAIt;
        if (MFMAIt == MBB.end() || !isMFMA(*MFMAIt) ||
            !isSimpleCrossableInstr(*Mid)) {
          errs() << "SIFixSBO[" << MF->getName() << " BB"
                 << MBB.getNumber() << "]: skip, next=" << Next->getOpcode()
                 << "\n";
          continue;
        }

        MachineInstr &MFMA = *MFMAIt;
        if (regsConflict(Perm, MFMA) || regsConflict(*Mid, MFMA))
          continue;

        auto PermIt = Perm.getIterator();
        MBB.splice(PermIt, &MBB, MFMA.getIterator());
        Changed = true;
        continue;
      }

      MachineInstr &MFMA = *Next;
      if (regsConflict(Perm, MFMA)) {
        continue;
      }

      auto PermIt = Perm.getIterator();
      auto MFMAIt = MFMA.getIterator();
      MBB.splice(PermIt, &MBB, MFMAIt);
      Changed = true;
    }
  }

  return Changed;
}

bool SIFixSchedBarrierOrder::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= processBlock(MBB, *TII);
    Changed |= hoistMFMAOverDSPermute(MBB);
  }

  return Changed;
}
