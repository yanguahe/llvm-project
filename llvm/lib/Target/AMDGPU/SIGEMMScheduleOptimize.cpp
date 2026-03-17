//===-- SIGEMMScheduleOptimize.cpp - GEMM loop schedule optimization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes FMHA-style GEMM loops by:
//   T1: Hoisting V global_loads before the first s_barrier
//   T2: Converting serial ds_read/MFMA chains to double-buffered (GEMM1)
//   T3: Interleaving V ds_writes into GEMM1 MFMA slots
//   T4: Double-buffering GEMM2 ds_read/MFMA chains (V^T reads)
//   T5: Defer V global loads past Barrier1 to overlap with GEMM1
//
// Runs before SIInsertWaitcnts so waitcnt values are auto-generated.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>

using namespace llvm;

#define DEBUG_TYPE "si-gemm-schedule-opt"

static cl::opt<bool> EnableGEMMScheduleOpt(
    "amdgpu-gemm-schedule-opt", cl::init(false), cl::Hidden,
    cl::desc("Enable GEMM loop schedule optimization"));

static cl::opt<unsigned> MinChainLength(
    "amdgpu-gemm-min-chain", cl::init(4), cl::Hidden,
    cl::desc("Minimum ds_read/MFMA chain length to trigger"));

namespace {

struct ReadMFMAPair {
  MachineInstr *Read;
  MachineInstr *MFMA;
};

struct VLoadInfo {
  MachineInstr *AddrCompute = nullptr;
  MachineInstr *Load = nullptr;
  Register LoadDest;
  SmallVector<MachineInstr *, 8> Writes;
};

class SIGEMMScheduleOptimize {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;

  static bool isDSRead2B32(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::DS_READ2_B32 || Opc == AMDGPU::DS_READ2_B32_gfx9;
  }

  static bool isDSWriteB16(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::DS_WRITE_B16 || Opc == AMDGPU::DS_WRITE_B16_gfx9 ||
           Opc == AMDGPU::DS_WRITE_B16_D16_HI ||
           Opc == AMDGPU::DS_WRITE_B16_D16_HI_vi ||
           Opc == AMDGPU::DS_WRITE_B16_t16;
  }

  static bool isGlobalLoadDwordX4(const MachineInstr &MI) {
    unsigned Opc = MI.getOpcode();
    return Opc == AMDGPU::GLOBAL_LOAD_DWORDX4 ||
           Opc == AMDGPU::GLOBAL_LOAD_DWORDX4_SADDR;
  }

  static bool isBufferLoadLDS(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    case AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN:
    case AMDGPU::BUFFER_LOAD_DWORD_LDS_IDXEN:
    case AMDGPU::BUFFER_LOAD_DWORD_LDS_BOTHEN:
    case AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFSET:
    case AMDGPU::BUFFER_LOAD_USHORT_LDS_OFFEN:
    case AMDGPU::BUFFER_LOAD_USHORT_LDS_IDXEN:
    case AMDGPU::BUFFER_LOAD_USHORT_LDS_BOTHEN:
    case AMDGPU::BUFFER_LOAD_USHORT_LDS_OFFSET:
    case AMDGPU::BUFFER_LOAD_UBYTE_LDS_OFFEN:
    case AMDGPU::BUFFER_LOAD_UBYTE_LDS_IDXEN:
    case AMDGPU::BUFFER_LOAD_UBYTE_LDS_BOTHEN:
    case AMDGPU::BUFFER_LOAD_UBYTE_LDS_OFFSET:
      return true;
    default:
      return false;
    }
  }

  bool findReadMFMAChain(MachineBasicBlock &MBB,
                         SmallVectorImpl<ReadMFMAPair> &Chain);
  bool findGEMM2Chain(MachineBasicBlock &MBB, MachineInstr *AfterBarrier,
                      SmallVectorImpl<ReadMFMAPair> &Chain);
  bool findBarrierPair(MachineBasicBlock &MBB,
                       const SmallVectorImpl<ReadMFMAPair> &Chain,
                       MachineInstr *&Barrier1, MachineInstr *&Barrier2);
  bool findVLoadWritePattern(MachineBasicBlock &MBB,
                             MachineInstr *Barrier1, MachineInstr *Barrier2,
                             VLoadInfo &VHi, VLoadInfo &VLo);

  Register findFreeVReg64(MachineBasicBlock &MBB);
  bool doubleBufferChain(MachineBasicBlock &MBB,
                         SmallVectorImpl<ReadMFMAPair> &Chain,
                         unsigned StartIdx);
  bool processLoop(MachineBasicBlock &MBB);

public:
  bool run(MachineFunction &MF);
};

class SIGEMMScheduleOptimizeLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIGEMMScheduleOptimizeLegacy() : MachineFunctionPass(ID) {
    initializeSIGEMMScheduleOptimizeLegacyPass(
        *PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return SIGEMMScheduleOptimize().run(MF);
  }
  StringRef getPassName() const override {
    return "SI GEMM Schedule Optimize";
  }
};

} // end anonymous namespace

INITIALIZE_PASS(SIGEMMScheduleOptimizeLegacy, DEBUG_TYPE,
                "SI GEMM schedule optimization", false, false)

char SIGEMMScheduleOptimizeLegacy::ID = 0;
char &llvm::SIGEMMScheduleOptimizeLegacyID =
    SIGEMMScheduleOptimizeLegacy::ID;

FunctionPass *llvm::createSIGEMMScheduleOptimizePass() {
  return new SIGEMMScheduleOptimizeLegacy();
}

// ===================================================================
// Pattern Recognition
// ===================================================================

bool SIGEMMScheduleOptimize::findReadMFMAChain(
    MachineBasicBlock &MBB, SmallVectorImpl<ReadMFMAPair> &Chain) {
  Chain.clear();
  Register ReadDstReg, ReadBaseReg;

  for (auto II = MBB.begin(), IE = MBB.end(); II != IE;) {
    MachineInstr &MI = *II;
    ++II;
    if (!isDSRead2B32(MI))
      continue;

    Register DstReg = MI.getOperand(0).getReg();
    Register BaseReg = MI.getOperand(1).getReg();

    if (Chain.empty()) {
      ReadDstReg = DstReg;
      ReadBaseReg = BaseReg;
    } else if (DstReg != ReadDstReg || BaseReg != ReadBaseReg) {
      break;
    }

    MachineInstr *PairedMFMA = nullptr;
    for (auto JJ = II; JJ != IE; ++JJ) {
      if (!SIInstrInfo::isMFMA(*JJ))
        continue;
      if (JJ->getOperand(1).isReg() &&
          JJ->getOperand(1).getReg() == ReadDstReg) {
        PairedMFMA = &*JJ;
        II = std::next(MachineBasicBlock::iterator(PairedMFMA));
        break;
      }
      break;
    }
    if (!PairedMFMA)
      break;
    Chain.push_back({&MI, PairedMFMA});
  }
  return Chain.size() >= MinChainLength;
}

bool SIGEMMScheduleOptimize::findBarrierPair(
    MachineBasicBlock &MBB, const SmallVectorImpl<ReadMFMAPair> &Chain,
    MachineInstr *&Barrier1, MachineInstr *&Barrier2) {
  Barrier1 = nullptr;
  Barrier2 = nullptr;

  MachineInstr *FirstRead = Chain.front().Read;
  MachineInstr *LastMFMA = Chain.back().MFMA;
  bool SeenFirstRead = false;
  bool SeenLastMFMA = false;

  for (auto &MI : MBB) {
    if (&MI == FirstRead) SeenFirstRead = true;
    if (&MI == LastMFMA) SeenLastMFMA = true;

    if (MI.getOpcode() != AMDGPU::S_BARRIER)
      continue;

    if (!Barrier1 && !SeenFirstRead) {
      Barrier1 = &MI;
      continue;
    }
    if (Barrier1 && SeenLastMFMA) {
      Barrier2 = &MI;
      break;
    }
  }
  return Barrier1 && Barrier2;
}

bool SIGEMMScheduleOptimize::findVLoadWritePattern(
    MachineBasicBlock &MBB, MachineInstr *Barrier1, MachineInstr *Barrier2,
    VLoadInfo &VHi, VLoadInfo &VLo) {

  // Find GLOBAL_LOAD_DWORDX4 instructions between Barrier1 and Barrier2.
  // In the original schedule they appear after softmax/O-rescale.
  SmallVector<MachineInstr *, 4> GlobalLoads;
  bool PastBarrier1 = false;
  for (auto &MI : MBB) {
    if (&MI == Barrier1) PastBarrier1 = true;
    if (&MI == Barrier2) break;
    if (PastBarrier1 && isGlobalLoadDwordX4(MI))
      GlobalLoads.push_back(&MI);
  }

  if (GlobalLoads.size() != 2)
    return false;

  // Assign: first load = VHi, second = VLo (matching script convention)
  VHi.Load = GlobalLoads[0];
  VLo.Load = GlobalLoads[1];
  VHi.LoadDest = VHi.Load->getOperand(0).getReg();
  VLo.LoadDest = VLo.Load->getOperand(0).getReg();

  // Find the address computation just before the first V load.
  // Walk backward from VHi.Load to find the first non-load instruction
  // that defines the address register used by VHi.Load.
  Register AddrReg = VHi.Load->getOperand(1).getReg();
  for (auto II = MachineBasicBlock::reverse_iterator(VHi.Load->getIterator()),
            IE = MBB.rend(); II != IE; ++II) {
    for (const auto &MO : II->operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg() == AddrReg) {
        VHi.AddrCompute = &*II;
        VLo.AddrCompute = nullptr; // shares same addr
        goto found_addr;
      }
    }
  }
found_addr:

  // Collect DS_WRITE_B16 instructions that source from V load destinations.
  // They appear between the V loads and Barrier2.
  auto getVGPR32Subs = [&](Register SuperReg) -> SmallVector<MCRegister, 4> {
    SmallVector<MCRegister, 4> Subs;
    for (MCRegister Sub : TRI->subregs(SuperReg.asMCReg())) {
      if (AMDGPU::VGPR_32RegClass.contains(Sub))
        Subs.push_back(Sub);
    }
    return Subs;
  };

  auto VHiSubs = getVGPR32Subs(VHi.LoadDest);
  auto VLoSubs = getVGPR32Subs(VLo.LoadDest);
  DenseSet<unsigned> VHiRegs(VHiSubs.begin(), VHiSubs.end());
  DenseSet<unsigned> VLoRegs(VLoSubs.begin(), VLoSubs.end());

  bool PastLoads = false;
  for (auto &MI : MBB) {
    if (&MI == VLo.Load) PastLoads = true;
    if (!PastLoads) continue;
    if (&MI == Barrier2) break;

    if (!isDSWriteB16(MI))
      continue;

    // The data source operand for DS_WRITE_B16 is operand 1 (after addr).
    // Check which V load group it belongs to.
    for (const auto &MO : MI.operands()) {
      if (!MO.isReg() || MO.isDef())
        continue;
      MCRegister Reg = MO.getReg().asMCReg();
      if (VHiRegs.count(Reg)) {
        VHi.Writes.push_back(&MI);
        break;
      }
      if (VLoRegs.count(Reg)) {
        VLo.Writes.push_back(&MI);
        break;
      }
      // Also check sub-registers
      for (MCRegister Sub : TRI->subregs(Reg)) {
        if (VHiRegs.count(Sub)) { VHi.Writes.push_back(&MI); goto next_mi; }
        if (VLoRegs.count(Sub)) { VLo.Writes.push_back(&MI); goto next_mi; }
      }
    }
    next_mi:;
  }

  return VHi.Writes.size() >= 2 && VLo.Writes.size() >= 2;
}

// Find the GEMM2 ds_read/MFMA chain starting after AfterBarrier.
// Unlike GEMM1, GEMM2 uses different base registers per read (precomputed
// V^T addresses), so we only require a consistent dest register.
bool SIGEMMScheduleOptimize::findGEMM2Chain(
    MachineBasicBlock &MBB, MachineInstr *AfterBarrier,
    SmallVectorImpl<ReadMFMAPair> &Chain) {
  Chain.clear();
  Register ReadDstReg;
  bool PastBarrier = false;

  for (auto II = MBB.begin(), IE = MBB.end(); II != IE;) {
    MachineInstr &MI = *II;
    if (&MI == AfterBarrier)
      PastBarrier = true;
    ++II;
    if (!PastBarrier)
      continue;
    if (!isDSRead2B32(MI))
      continue;

    Register DstReg = MI.getOperand(0).getReg();
    if (Chain.empty()) {
      ReadDstReg = DstReg;
    } else if (DstReg != ReadDstReg) {
      break;
    }

    MachineInstr *PairedMFMA = nullptr;
    for (auto JJ = II; JJ != IE; ++JJ) {
      if (!SIInstrInfo::isMFMA(*JJ))
        continue;
      if (JJ->getOperand(1).isReg() &&
          JJ->getOperand(1).getReg() == ReadDstReg) {
        PairedMFMA = &*JJ;
        II = std::next(MachineBasicBlock::iterator(PairedMFMA));
        break;
      }
      break;
    }
    if (!PairedMFMA)
      break;
    Chain.push_back({&MI, PairedMFMA});
  }
  return Chain.size() >= MinChainLength;
}

Register SIGEMMScheduleOptimize::findFreeVReg64(MachineBasicBlock &MBB) {
  DenseSet<unsigned> UsedVGPR32;
  for (auto &MI : MBB) {
    for (const auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      MCRegister Reg = MO.getReg().asMCReg();
      if (AMDGPU::VGPR_32RegClass.contains(Reg))
        UsedVGPR32.insert(Reg);
      for (MCRegister Sub : TRI->subregs(Reg))
        if (AMDGPU::VGPR_32RegClass.contains(Sub))
          UsedVGPR32.insert(Sub);
    }
  }
  for (const auto &LI : MBB.liveins()) {
    MCRegister Reg = LI.PhysReg;
    if (AMDGPU::VGPR_32RegClass.contains(Reg))
      UsedVGPR32.insert(Reg);
    for (MCRegister Sub : TRI->subregs(Reg))
      if (AMDGPU::VGPR_32RegClass.contains(Sub))
        UsedVGPR32.insert(Sub);
  }
  for (unsigned R = 0; R < AMDGPU::VReg_64RegClass.getNumRegs(); ++R) {
    MCRegister P = AMDGPU::VReg_64RegClass.getRegister(R);
    bool Free = true;
    for (MCRegister Sub : TRI->subregs(P))
      if (AMDGPU::VGPR_32RegClass.contains(Sub) && UsedVGPR32.count(Sub)) {
        Free = false;
        break;
      }
    if (Free)
      return P;
  }
  return Register();
}

// Apply double-buffering to a ds_read/MFMA chain starting at index StartIdx.
// Reorders reads so that Read[i+1] is issued before MFMA[i], and alternates
// the dest register between BufA (original) and BufB (free register).
bool SIGEMMScheduleOptimize::doubleBufferChain(
    MachineBasicBlock &MBB, SmallVectorImpl<ReadMFMAPair> &Chain,
    unsigned StartIdx) {
  unsigned N = Chain.size();
  if (StartIdx >= N || N - StartIdx < 2)
    return false;

  Register BufA = Chain[StartIdx].Read->getOperand(0).getReg();
  Register BufB = findFreeVReg64(MBB);
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[T4-DB] N=" << N << " StartIdx=" << StartIdx
        << " BufA=" << TRI->getName(BufA)
        << " BufB=" << (BufB ? TRI->getName(BufB) : "NONE")
        << "\n";
    for (unsigned i = 0; i < N; ++i) {
      log << "  [" << i << "] Read dst=" << TRI->getName(Chain[i].Read->getOperand(0).getReg())
          << " MFMA srcA=" << TRI->getName(Chain[i].MFMA->getOperand(1).getReg())
          << "\n";
    }
  }
  if (!BufB)
    return false;

  // Move Read[StartIdx+1] before MFMA[StartIdx], set dest to BufB
  if (StartIdx + 1 < N) {
    MachineInstr *R1 = Chain[StartIdx + 1].Read;
    R1->removeFromParent();
    MBB.insert(MachineBasicBlock::iterator(Chain[StartIdx].MFMA), R1);
    R1->getOperand(0).setReg(BufB);
  }

  // For StartIdx+1..N-2: move Read[i+1] before MFMA[i], alternate bufs
  for (unsigned i = StartIdx + 1; i < N - 1; ++i) {
    unsigned relIdx = i - StartIdx;
    bool UseB = (relIdx % 2 == 1);
    Register UseBuf = UseB ? BufB : BufA;
    Register PrefBuf = UseB ? BufA : BufB;

    Chain[i].MFMA->getOperand(1).setReg(UseBuf);

    MachineInstr *NextRead = Chain[i + 1].Read;
    NextRead->removeFromParent();
    MBB.insert(MachineBasicBlock::iterator(Chain[i].MFMA), NextRead);
    NextRead->getOperand(0).setReg(PrefBuf);
  }

  // Last MFMA uses the last-written buffer
  {
    unsigned relIdx = (N - 1) - StartIdx;
    bool UseB = (relIdx % 2 == 1);
    Chain[N - 1].MFMA->getOperand(1).setReg(UseB ? BufB : BufA);
  }
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[T4-DB] After transform:\n";
    for (unsigned i = 0; i < N; ++i) {
      log << "  [" << i << "] Read dst=" << TRI->getName(Chain[i].Read->getOperand(0).getReg())
          << " MFMA srcA=" << TRI->getName(Chain[i].MFMA->getOperand(1).getReg())
          << "\n";
    }
  }
  return true;
}

// ===================================================================
// Transformations
// ===================================================================

bool SIGEMMScheduleOptimize::processLoop(MachineBasicBlock &MBB) {
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] processLoop entered, MBB=" << MBB.getName().str() << "\n";
  }

  // Skip when async buffer_load→LDS (DMA) is detected.  The new CK-aligned
  // pipeline manages its own K prefetch via buffer_load_dword...lds;
  // T1-T5 transformations are not applicable and could corrupt the schedule.
  for (const MachineInstr &MI : MBB) {
    if (isBufferLoadLDS(MI)) {
      std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
      log << "[GEMMOpt] Async buffer_load→LDS detected, skipping MBB\n";
      return false;
    }
  }

  // Step 1: Find GEMM1 chain
  SmallVector<ReadMFMAPair, 32> Chain;
  if (!findReadMFMAChain(MBB, Chain)) {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] findReadMFMAChain FAILED, chain_size=" << Chain.size() << "\n";
    return false;
  }

  unsigned N = Chain.size();
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] GEMM1 chain found, N=" << N << "\n";
  }

  MachineInstr *Barrier1 = nullptr, *Barrier2 = nullptr;
  if (!findBarrierPair(MBB, Chain, Barrier1, Barrier2)) {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] findBarrierPair FAILED\n";
    return false;
  }
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] barriers found\n";
  }

  // ===== T5: Sink V global loads to just before Barrier1 =====
  // In the original schedule, V and K global loads are interleaved:
  //   K1, V1, V2, ds_write(K1), K2, ds_write(K1), ds_write(K2)...
  // This forces vmcnt to wait for V loads even though their data isn't
  // needed until after GEMM1.  By sinking V loads to just before
  // Barrier1, the vmcnt for K ds_writes only counts K loads.  V loads
  // remain in-flight through the barrier and GEMM1, hiding HBM latency
  // (gfx942 BackOffBarrier: no vmcnt(0) inserted before S_BARRIER).
  {
    SmallVector<MachineInstr *, 4> VLoadsToSink;
    for (auto II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      if (&*II == Barrier1)
        break;
      if (!isGlobalLoadDwordX4(*II))
        continue;

      Register Dst = II->getOperand(0).getReg();
      bool ConsumedBeforeBarrier = false;
      for (auto JJ = std::next(II); JJ != IE; ++JJ) {
        if (&*JJ == Barrier1)
          break;
        for (const MachineOperand &MO : JJ->operands()) {
          if (MO.isReg() && MO.isUse() && TRI->regsOverlap(MO.getReg(), Dst)) {
            ConsumedBeforeBarrier = true;
            break;
          }
        }
        if (ConsumedBeforeBarrier)
          break;
      }
      if (!ConsumedBeforeBarrier)
        VLoadsToSink.push_back(&*II);
    }

    if (!VLoadsToSink.empty()) {
      auto InsertPt = MachineBasicBlock::iterator(Barrier1);
      for (MachineInstr *VL : VLoadsToSink) {
        VL->removeFromParent();
        MBB.insert(InsertPt, VL);
      }
      std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
      log << "[GEMMOpt-T5] Sank " << VLoadsToSink.size()
          << " V global loads to just before Barrier1\n";
    } else {
      std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
      log << "[GEMMOpt-T5] No sinkable V global loads found\n";
    }
  }

  VLoadInfo VHi, VLo;
  bool HasVLoadPattern = findVLoadWritePattern(MBB, Barrier1, Barrier2, VHi, VLo);
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] VLoad pattern: " << HasVLoadPattern << "\n";
  }

  unsigned DBStart = 0;
  Register SecondBuf;

  if (HasVLoadPattern) {
    unsigned VHiWritesPerSlot = 2;
    unsigned VLoWritesPerSlot = 2;
    unsigned VHiSlots = std::min((unsigned)VHi.Writes.size() / VHiWritesPerSlot,
                                 std::min(N, 4u));
    unsigned VLoSlots = std::min((unsigned)VLo.Writes.size() / VLoWritesPerSlot,
                                 std::min(N - VHiSlots, 4u));
    DBStart = VHiSlots;
    unsigned VLoStart = N > (VLoSlots + 2) ? N - VLoSlots - 2 : DBStart;

    // ===== T1: Hoist V loads before Barrier1 =====
    auto BarrierIt = MachineBasicBlock::iterator(Barrier1);
    if (VHi.AddrCompute) {
      VHi.AddrCompute->removeFromParent();
      MBB.insert(BarrierIt, VHi.AddrCompute);
    }
    VHi.Load->removeFromParent();
    MBB.insert(BarrierIt, VHi.Load);
    VLo.Load->removeFromParent();
    MBB.insert(BarrierIt, VLo.Load);

    // ===== T3a: Interleave V hi writes into GEMM1 #0..VHiSlots-1 =====
    unsigned vhiIdx = 0;
    for (unsigned i = 0; i < VHiSlots && vhiIdx < VHi.Writes.size(); ++i) {
      auto InsertPt = MachineBasicBlock::iterator(Chain[i].MFMA);
      for (unsigned w = 0; w < VHiWritesPerSlot && vhiIdx < VHi.Writes.size();
           ++w, ++vhiIdx) {
        MachineInstr *VW = VHi.Writes[vhiIdx];
        VW->removeFromParent();
        MBB.insert(InsertPt, VW);
      }
    }

    // SecondBuf from VHi.LoadDest sub-register
    SmallVector<MCRegister, 4> HiSubs;
    for (MCRegister Sub : TRI->subregs(VHi.LoadDest.asMCReg())) {
      if (AMDGPU::VGPR_32RegClass.contains(Sub))
        HiSubs.push_back(Sub);
    }
    if (HiSubs.size() >= 2) {
      for (MCRegister Super : TRI->superregs(HiSubs[0])) {
        if (AMDGPU::VReg_64RegClass.contains(Super) &&
            TRI->isSubRegisterEq(Super, HiSubs[1])) {
          SecondBuf = Super;
          break;
        }
      }
    }

    // ===== T3b: Interleave V lo writes into GEMM1 tail =====
    unsigned vloIdx = 0;
    for (unsigned i = 0; i < VLoSlots && vloIdx < VLo.Writes.size(); ++i) {
      unsigned chainIdx = VLoStart + i;
      if (chainIdx >= N) break;
      auto InsertPt = MachineBasicBlock::iterator(Chain[chainIdx].MFMA);
      for (unsigned w = 0; w < VLoWritesPerSlot && vloIdx < VLo.Writes.size();
           ++w, ++vloIdx) {
        MachineInstr *VW = VLo.Writes[vloIdx];
        VW->removeFromParent();
        MBB.insert(InsertPt, VW);
      }
    }
  }

  // ===== T2: K double-buffer for GEMM1 #DBStart..N-1 =====
  if (!SecondBuf)
    SecondBuf = findFreeVReg64(MBB);

  if (SecondBuf) {
    Register BufA = Chain[0].Read->getOperand(0).getReg();
    Register BufB = SecondBuf;

    if (DBStart < N && N - DBStart >= 2) {
      if (DBStart + 1 < N) {
        MachineInstr *R1 = Chain[DBStart + 1].Read;
        R1->removeFromParent();
        MBB.insert(MachineBasicBlock::iterator(Chain[DBStart].MFMA), R1);
        R1->getOperand(0).setReg(BufB);
      }

      for (unsigned i = DBStart + 1; i < N - 1; ++i) {
        unsigned relIdx = i - DBStart;
        bool UseB = (relIdx % 2 == 1);
        Register UseBuf = UseB ? BufB : BufA;
        Register PrefBuf = UseB ? BufA : BufB;

        Chain[i].MFMA->getOperand(1).setReg(UseBuf);

        MachineInstr *NextRead = Chain[i + 1].Read;
        NextRead->removeFromParent();
        MBB.insert(MachineBasicBlock::iterator(Chain[i].MFMA), NextRead);
        NextRead->getOperand(0).setReg(PrefBuf);
      }

      unsigned relIdx = (N - 1) - DBStart;
      bool UseB = (relIdx % 2 == 1);
      Chain[N - 1].MFMA->getOperand(1).setReg(UseB ? BufB : BufA);
    }
  }

  // ===== T4: Double-buffer GEMM2 (V^T read / MFMA chain after Barrier2) =====
  SmallVector<ReadMFMAPair, 32> G2Chain;
  bool FoundG2 = findGEMM2Chain(MBB, Barrier2, G2Chain);
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt-T4] GEMM2 chain: found=" << FoundG2
        << " size=" << G2Chain.size() << "\n";
  }
  if (FoundG2) {
    bool DB = doubleBufferChain(MBB, G2Chain, 0);
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt-T4] doubleBufferChain result=" << DB << "\n";
    // Dump GEMM2 instructions after transform
    log << "[GEMMOpt-T4] Post-T4 GEMM2 section:\n";
    bool inGemm2 = false;
    for (auto II = MBB.begin(), IE = MBB.end(); II != IE; ++II) {
      if (&*II == Barrier2) inGemm2 = true;
      if (!inGemm2) continue;
      std::string buf;
      raw_string_ostream os(buf);
      II->print(os);
      log << "  " << os.str();
      if (II->getOpcode() == AMDGPU::S_CBRANCH_VCCNZ ||
          II->getOpcode() == AMDGPU::S_BRANCH)
        break;
    }
  }

  return true;
}

bool SIGEMMScheduleOptimize::run(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  Attribute GSOAttr = F.getFnAttribute("amdgpu-gemm-schedule-opt");
  bool Enable = EnableGEMMScheduleOpt;
  if (GSOAttr.isValid())
    Enable = (GSOAttr.getValueAsString() != "false");
  {
    std::ofstream log("/tmp/gemmopt_diag.txt", std::ios::app);
    log << "[GEMMOpt] run() Enable=" << Enable
        << " fn=" << F.getName().str() << "\n";
  }
  if (!Enable)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool HasBackedge = false;
    for (MachineBasicBlock *Succ : MBB.successors())
      if (Succ == &MBB) { HasBackedge = true; break; }
    if (!HasBackedge)
      continue;
    Changed |= processLoop(MBB);
  }
  return Changed;
}
