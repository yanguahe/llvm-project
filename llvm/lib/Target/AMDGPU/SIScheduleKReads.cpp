//===- SIScheduleKReads.cpp - Interleave ds_read_b128 with MFMA -----------===//
//
// Post-RA pass that finds a batch of 4+ DS_READ_B128 (K reads) followed
// by a sequence of MFMAs, and distributes the reads among the MFMAs.
//
// Input pattern (after sched_barrier batching):
//   8×DS_READ_B128
//   SCHED_BARRIER
//   MFMA_0, [valu...], MFMA_1, [valu...], ..., MFMA_15, [valu/ds_read_b64...]
//
// Output pattern:
//   DS_READ_B128_0
//   MFMA_0, [valu...], MFMA_1, [valu...]
//   DS_READ_B128_1
//   MFMA_2, [valu...], MFMA_3, [valu...]
//   ...
//
// This gives 32+ cycle gaps between consecutive LDS reads.
// Runs before SIInsertWaitcnts.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "si-schedule-kreads"

static cl::opt<bool> EnableScheduleKReads(
    "amdgpu-schedule-kreads",
    cl::desc("Interleave DS_READ_B128 groups with MFMA for 32+ cycle spacing"),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> MFMAPerRead(
    "amdgpu-mfma-per-read",
    cl::desc("Number of MFMA instructions between ds_read groups"),
    cl::init(2), cl::Hidden);

namespace {

class SIScheduleKReads : public MachineFunctionPass {
public:
  static char ID;
  SIScheduleKReads() : MachineFunctionPass(ID) {
    initializeSIScheduleKReadsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "SI Schedule K Reads";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char SIScheduleKReads::ID = 0;

INITIALIZE_PASS(SIScheduleKReads, DEBUG_TYPE,
                "SI Schedule K Reads", false, false)

FunctionPass *llvm::createSIScheduleKReadsPass() {
  return new SIScheduleKReads();
}

static bool isDSReadB128(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == AMDGPU::DS_READ_B128 || Opc == AMDGPU::DS_READ_B128_gfx9;
}

bool SIScheduleKReads::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableScheduleKReads)
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Phase 1: Find batched DS_READ_B128 groups.
    SmallVector<MachineInstr *, 8> DSReads;

    auto It = MBB.begin();
    while (It != MBB.end()) {
      if (!isDSReadB128(*It)) {
        ++It;
        continue;
      }

      DSReads.clear();
      auto ReadStart = It;
      while (It != MBB.end() && isDSReadB128(*It)) {
        DSReads.push_back(&*It);
        ++It;
      }

      if (DSReads.size() < 4)
        continue;

      // Skip SCHED_BARRIER and other pseudo instructions.
      while (It != MBB.end() &&
             (It->getOpcode() == AMDGPU::SCHED_BARRIER ||
              It->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER ||
              It->isTransient())) {
        ++It;
      }

      // Phase 2: Find MFMAs in the subsequent instruction stream.
      // Count MFMAs and record their positions.
      SmallVector<MachineInstr *, 16> MFMAPositions;
      auto ScanIt = It;
      while (ScanIt != MBB.end()) {
        if (ScanIt->getOpcode() == AMDGPU::S_BARRIER)
          break;
        if (SIInstrInfo::isMFMA(*ScanIt))
          MFMAPositions.push_back(&*ScanIt);
        ++ScanIt;
      }

      unsigned NumReads = DSReads.size();
      unsigned NumMFMAs = MFMAPositions.size();
      unsigned Ratio = MFMAPerRead;

      if (NumMFMAs < NumReads * Ratio || NumMFMAs < 8)
        continue;

      LLVM_DEBUG(dbgs() << "SIScheduleKReads: distributing " << NumReads
                        << " DS_READ_B128 among " << NumMFMAs << " MFMAs"
                        << " (ratio=" << Ratio << ")\n");

      // Phase 3: Remove all ds_reads from their batch position.
      for (MachineInstr *MI : DSReads)
        MI->removeFromParent();

      // Remove SCHED_BARRIER between reads and MFMAs (no longer needed).
      auto CleanIt = ReadStart;
      while (CleanIt != MBB.end() &&
             (CleanIt->getOpcode() == AMDGPU::SCHED_BARRIER ||
              CleanIt->getOpcode() == AMDGPU::SCHED_GROUP_BARRIER)) {
        auto ToRemove = CleanIt++;
        ToRemove->removeFromParent();
      }

      // Phase 4: Insert each ds_read before every Ratio-th MFMA.
      for (unsigned i = 0; i < NumReads; ++i) {
        unsigned TargetMFMA = i * Ratio;
        if (TargetMFMA < NumMFMAs) {
          MBB.insert(MachineBasicBlock::iterator(MFMAPositions[TargetMFMA]),
                     DSReads[i]);
        } else {
          // Place remaining reads before the last MFMA.
          MBB.insert(
              MachineBasicBlock::iterator(MFMAPositions[NumMFMAs - 1]),
              DSReads[i]);
        }
      }

      Changed = true;
      // Restart scanning this block.
      It = MBB.begin();
    }
  }

  return Changed;
}
