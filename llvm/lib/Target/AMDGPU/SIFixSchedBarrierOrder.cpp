//===- SIFixSchedBarrierOrder.cpp - Consolidate MFMAs before buffer_loads -===//
//
// Post-hazard-recognizer pass:
// 1. Separates V-prefetch buffer_loads from GEMM1 MFMA chains.
// 2. Ensures exactly 1 MFMA before each ds_permute_b32 (matching reference
//    ASM pattern). Uses register renaming to resolve WAR conflicts when
//    the scheduler places 2+ MFMAs before ds_permute.
//
//===----------------------------------------------------------------------===//

#include "SIFixSchedBarrierOrder.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sched-barrier-order"

namespace {

// Core implementation shared by legacy and new pass manager wrappers.
class SIFixSchedBarrierOrderImpl {
public:
  bool run(MachineFunction &MF);

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

  bool sinkBreakersFromMFMAChain(MachineBasicBlock &MBB);

  bool coalescePkFmaDstSrc0(MachineBasicBlock &MBB, const SIInstrInfo &TII);

  bool tryRenameDSPermuteDst(MachineBasicBlock &MBB, MachineInstr &Perm,
                             MachineInstr &MFMAToMove,
                             MachineBasicBlock::iterator InsertPt);

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

// Legacy pass manager wrapper.
class SIFixSchedBarrierOrderLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIFixSchedBarrierOrderLegacy() : MachineFunctionPass(ID) {
    initializeSIFixSchedBarrierOrderLegacyPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "SI Fix SCHED_BARRIER Ordering";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    return SIFixSchedBarrierOrderImpl().run(MF);
  }
};

} // end anonymous namespace

char SIFixSchedBarrierOrderLegacy::ID = 0;

INITIALIZE_PASS(SIFixSchedBarrierOrderLegacy, DEBUG_TYPE,
                "SI Fix SCHED_BARRIER Ordering", false, false)

FunctionPass *llvm::createSIFixSchedBarrierOrderPass() {
  return new SIFixSchedBarrierOrderLegacy();
}

// New pass manager entry point.
PreservedAnalyses
SIFixSchedBarrierOrderPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  if (!SIFixSchedBarrierOrderImpl().run(MF))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

//===----------------------------------------------------------------------===//
// Core implementation
//===----------------------------------------------------------------------===//

bool SIFixSchedBarrierOrderImpl::processBlock(MachineBasicBlock &MBB,
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

bool SIFixSchedBarrierOrderImpl::tryRenameDSPermuteDst(
    MachineBasicBlock &MBB, MachineInstr &Perm, MachineInstr &MFMAToMove,
    MachineBasicBlock::iterator InsertPt) {

  Register PermDst;
  for (const MachineOperand &MO : Perm.operands()) {
    if (MO.isReg() && MO.isDef()) {
      PermDst = MO.getReg();
      break;
    }
  }
  if (!PermDst.isValid() || !PermDst.isPhysical())
    return false;

  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  bool HasWAR = false;
  for (const MachineOperand &MO : MFMAToMove.operands()) {
    if (MO.isReg() && MO.isUse() && TRI->regsOverlap(MO.getReg(), PermDst)) {
      HasWAR = true;
      break;
    }
  }
  if (!HasWAR)
    return false;

  MachineInstr *FirstUse = nullptr;
  for (auto FIt = std::next(Perm.getIterator()); FIt != MBB.end(); ++FIt) {
    for (const MachineOperand &MO : FIt->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg() == PermDst) {
        FirstUse = &*FIt;
        goto found_use;
      }
    }
  }
found_use:
  if (!FirstUse)
    return false;

  MachineInstr *ClosestMFMA = nullptr;
  {
    auto Prev = Perm.getIterator();
    if (Prev != MBB.begin()) {
      --Prev;
      if (isMFMA(*Prev))
        ClosestMFMA = &*Prev;
    }
  }
  if (!ClosestMFMA)
    return false;

  SmallVector<MCPhysReg, 8> Candidates;
  for (const MachineOperand &MO : ClosestMFMA->operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isPhysical())
      continue;
    if (!TRI->isVGPR(MRI, MO.getReg()))
      continue;
    MCPhysReg R = MO.getReg().asMCReg();
    for (MCSubRegIterator Sub(R, TRI, /*IncludeSelf=*/true); Sub.isValid();
         ++Sub) {
      if (AMDGPU::VGPR_32RegClass.contains(*Sub))
        Candidates.push_back(*Sub);
    }
  }

  MCPhysReg FreeReg = AMDGPU::NoRegister;
  for (MCPhysReg Cand : Candidates) {
    if (Cand == PermDst.asMCReg())
      continue;

    bool Conflict = false;
    for (auto ChkIt = std::next(Perm.getIterator());
         ChkIt != std::next(FirstUse->getIterator()); ++ChkIt) {
      for (const MachineOperand &MO : ChkIt->operands()) {
        if (!MO.isReg())
          continue;
        Register R = MO.getReg();
        if (&*ChkIt == FirstUse && R == PermDst)
          continue;
        if (TRI->regsOverlap(R, Cand)) {
          Conflict = true;
          break;
        }
      }
      if (Conflict)
        break;
    }
    if (Conflict)
      continue;

    bool ConflictsWithMFMA = false;
    for (const MachineOperand &MO : MFMAToMove.operands()) {
      if (MO.isReg() && TRI->regsOverlap(Cand, MO.getReg())) {
        ConflictsWithMFMA = true;
        break;
      }
    }
    if (ConflictsWithMFMA)
      continue;

    FreeReg = Cand;
    break;
  }

  if (FreeReg == AMDGPU::NoRegister)
    return false;

  for (MachineOperand &MO : Perm.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == PermDst)
      MO.setReg(FreeReg);
  }

  for (auto RenIt = std::next(Perm.getIterator()); RenIt != MBB.end();
       ++RenIt) {
    bool FoundDef = false;
    for (const MachineOperand &MO : RenIt->operands()) {
      if (MO.isReg() && MO.isDef() &&
          TRI->regsOverlap(MO.getReg(), PermDst)) {
        FoundDef = true;
        break;
      }
    }
    for (MachineOperand &MO : RenIt->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg() == PermDst)
        MO.setReg(FreeReg);
    }
    if (FoundDef)
      break;
  }

  MBB.splice(InsertPt, &MBB, MFMAToMove);
  return true;
}

bool SIFixSchedBarrierOrderImpl::hoistMFMAOverDSPermute(
    MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto It = MBB.begin(); It != MBB.end(); ++It) {
    if (!isDSPermute(*It))
      continue;

    MachineInstr &Perm = *It;

    // Collect consecutive MFMAs immediately before ds_permute,
    // skipping SCHED_BARRIER pseudo-instructions (removed during emission).
    SmallVector<MachineInstr *, 4> MFMAsBefore;
    {
      auto Scan = It;
      while (Scan != MBB.begin()) {
        --Scan;
        if (isMFMA(*Scan))
          MFMAsBefore.push_back(&*Scan);
        else if (Scan->getOpcode() == AMDGPU::SCHED_BARRIER)
          continue; // skip SCHED_BARRIERs
        else
          break;
      }
    }

    if (MFMAsBefore.size() > 1) {
      auto IPt = std::next(It);
      for (int i = MFMAsBefore.size() - 1; i >= 1; --i) {
        MachineInstr *MI = MFMAsBefore[i];
        if (!regsConflict(Perm, *MI)) {
          MBB.splice(IPt, &MBB, MI);
          Changed = true;
        } else {
          Changed |= tryRenameDSPermuteDst(MBB, Perm, *MI, IPt);
        }
      }
      continue;
    }

    if (MFMAsBefore.size() == 1)
      continue;

    // No MFMA before ds_permute: hoist one from after.
    auto Next = std::next(It);
    while (Next != MBB.end() && isNopOrWaitcnt(*Next))
      ++Next;
    if (Next == MBB.end() || !isMFMA(*Next))
      continue;

    MachineInstr &MFMA = *Next;
    if (regsConflict(Perm, MFMA))
      continue;

    MBB.splice(Perm.getIterator(), &MBB, MFMA.getIterator());
    Changed = true;
  }

  return Changed;
}

static bool isRegAllocArtifact(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::SCRATCH_LOAD_DWORD:
  case AMDGPU::SCRATCH_LOAD_DWORD_SADDR:
  case AMDGPU::SCRATCH_LOAD_DWORDX2:
  case AMDGPU::SCRATCH_LOAD_DWORDX2_SADDR:
  case AMDGPU::SCRATCH_STORE_DWORD:
  case AMDGPU::SCRATCH_STORE_DWORD_SADDR:
  case AMDGPU::SCRATCH_STORE_DWORDX2:
  case AMDGPU::SCRATCH_STORE_DWORDX2_SADDR:
    return true;
  default:
    return false;
  }
}

bool SIFixSchedBarrierOrderImpl::sinkBreakersFromMFMAChain(
    MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto It = MBB.begin(); It != MBB.end();) {
    if (!isMFMA(*It)) {
      ++It;
      continue;
    }

    struct ChainEntry {
      MachineInstr *MI;
      bool IsMFMA;
    };
    SmallVector<ChainEntry, 32> Chain;
    auto Scan = It;

    while (Scan != MBB.end()) {
      if (isMFMA(*Scan)) {
        Chain.push_back({&*Scan, true});
        ++Scan;
      } else if (isRegAllocArtifact(*Scan)) {
        Chain.push_back({&*Scan, false});
        ++Scan;
      } else {
        break;
      }
    }

    unsigned MFMACount = 0, BreakerCount = 0;
    for (auto &E : Chain) {
      if (E.IsMFMA)
        MFMACount++;
      else
        BreakerCount++;
    }

    if (BreakerCount == 0 || MFMACount < 3) {
      It = Scan;
      continue;
    }

    MachineInstr *LastMFMA = nullptr;
    for (auto &E : Chain)
      if (E.IsMFMA)
        LastMFMA = E.MI;

    auto SinkPt = std::next(LastMFMA->getIterator());

    for (unsigned i = 0; i < Chain.size(); ++i) {
      if (Chain[i].IsMFMA)
        continue;

      bool HasMFMAAfter = false;
      for (unsigned j = i + 1; j < Chain.size(); ++j) {
        if (Chain[j].IsMFMA) {
          HasMFMAAfter = true;
          break;
        }
      }
      if (!HasMFMAAfter)
        continue;

      MachineInstr *Breaker = Chain[i].MI;
      bool CanSink = true;
      for (unsigned j = i + 1; j < Chain.size(); ++j) {
        if (!Chain[j].IsMFMA)
          continue;
        if (regsConflict(*Breaker, *Chain[j].MI)) {
          CanSink = false;
          break;
        }
      }

      if (CanSink) {
        MBB.splice(SinkPt, &MBB, Breaker);
        Changed = true;
      }
    }

    It = Scan;
  }

  return Changed;
}

bool SIFixSchedBarrierOrderImpl::coalescePkFmaDstSrc0(
    MachineBasicBlock &MBB, const SIInstrInfo &TII) {
  bool Changed = false;

  for (auto It = MBB.begin(); It != MBB.end(); ++It) {
    MachineInstr &MI = *It;
    if (MI.getOpcode() != AMDGPU::V_PK_FMA_F32)
      continue;

    Register Dst = MI.getOperand(0).getReg();
    int Src0Idx =
        AMDGPU::getNamedOperandIdx(AMDGPU::V_PK_FMA_F32, AMDGPU::OpName::src0);
    int Src2Idx =
        AMDGPU::getNamedOperandIdx(AMDGPU::V_PK_FMA_F32, AMDGPU::OpName::src2);
    Register Src0 = MI.getOperand(Src0Idx).getReg();
    Register Src2 = MI.getOperand(Src2Idx).getReg();

    if (Dst == Src0)
      continue;

    // If dst overlaps with src2, the copies would destroy src2's value.
    // Use temp registers (VGPR252_VGPR253) to save src2 first.
    bool DstOverlapsSrc2 =
        Src2.isPhysical() && TRI->regsOverlap(Dst, Src2);
    if (DstOverlapsSrc2) {
      Register TempPair = AMDGPU::VGPR252_VGPR253;
      Register TempLo = AMDGPU::VGPR252;
      Register TempHi = AMDGPU::VGPR253;
      Register Src2Lo = TRI->getSubReg(Src2, AMDGPU::sub0);
      Register Src2Hi = TRI->getSubReg(Src2, AMDGPU::sub1);

      BuildMI(MBB, It, MI.getDebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32),
              TempLo)
          .addReg(Src2Lo);
      BuildMI(MBB, It, MI.getDebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32),
              TempHi)
          .addReg(Src2Hi);
      MI.getOperand(Src2Idx).setReg(TempPair);
    }

    Register DstLo = TRI->getSubReg(Dst, AMDGPU::sub0);
    Register DstHi = TRI->getSubReg(Dst, AMDGPU::sub1);
    Register Src0Lo = TRI->getSubReg(Src0, AMDGPU::sub0);
    Register Src0Hi = TRI->getSubReg(Src0, AMDGPU::sub1);

    BuildMI(MBB, It, MI.getDebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32), DstLo)
        .addReg(Src0Lo);
    BuildMI(MBB, It, MI.getDebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32), DstHi)
        .addReg(Src0Hi);
    MI.getOperand(Src0Idx).setReg(Dst);
    Changed = true;
  }

  return Changed;
}

bool SIFixSchedBarrierOrderImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= processBlock(MBB, *TII);
    Changed |= hoistMFMAOverDSPermute(MBB);
    Changed |= sinkBreakersFromMFMAChain(MBB);
    Changed |= coalescePkFmaDstSrc0(MBB, *TII);
  }

  return Changed;
}
