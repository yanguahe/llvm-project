//===-- GCNSchedStrategy.cpp - GCN Scheduler Strategy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This contains a MachineSchedStrategy implementation for maximizing wave
/// occupancy on GCN hardware.
///
/// This pass will apply multiple scheduling stages to the same function.
/// Regions are first recorded in GCNScheduleDAGMILive::schedule. The actual
/// entry point for the scheduling of those regions is
/// GCNScheduleDAGMILive::runSchedStages.

/// Generally, the reason for having multiple scheduling stages is to account
/// for the kernel-wide effect of register usage on occupancy.  Usually, only a
/// few scheduling regions will have register pressure high enough to limit
/// occupancy for the kernel, so constraints can be relaxed to improve ILP in
/// other regions.
///
//===----------------------------------------------------------------------===//

#include "GCNSchedStrategy.h"
#include "AMDGPUIGroupLP.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<bool> DisableUnclusterHighRP(
    "amdgpu-disable-unclustered-high-rp-reschedule", cl::Hidden,
    cl::desc("Disable unclustered high register pressure "
             "reduction scheduling stage."),
    cl::init(false));

static cl::opt<bool> DisableClusteredLowOccupancy(
    "amdgpu-disable-clustered-low-occupancy-reschedule", cl::Hidden,
    cl::desc("Disable clustered low occupancy "
             "rescheduling for ILP scheduling stage."),
    cl::init(false));

static cl::opt<unsigned> ScheduleMetricBias(
    "amdgpu-schedule-metric-bias", cl::Hidden,
    cl::desc(
        "Sets the bias which adds weight to occupancy vs latency. Set it to "
        "100 to chase the occupancy only."),
    cl::init(10));

static cl::opt<bool>
    RelaxedOcc("amdgpu-schedule-relaxed-occupancy", cl::Hidden,
               cl::desc("Relax occupancy targets for kernels which are memory "
                        "bound (amdgpu-membound-threshold), or "
                        "Wave Limited (amdgpu-limit-wave-threshold)."),
               cl::init(false));

static cl::opt<bool> GCNTrackers(
    "amdgpu-use-amdgpu-trackers", cl::Hidden,
    cl::desc("Use the AMDGPU specific RPTrackers during scheduling"),
    cl::init(false));

static cl::opt<unsigned> PendingQueueLimit(
    "amdgpu-scheduler-pending-queue-limit", cl::Hidden,
    cl::desc(
        "Max (Available+Pending) size to inspect pending queue (0 disables)"),
    cl::init(256));

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define DUMP_MAX_REG_PRESSURE
static cl::opt<bool> PrintMaxRPRegUsageBeforeScheduler(
    "amdgpu-print-max-reg-pressure-regusage-before-scheduler", cl::Hidden,
    cl::desc("Print a list of live registers along with their def/uses at the "
             "point of maximum register pressure before scheduling."),
    cl::init(false));

static cl::opt<bool> PrintMaxRPRegUsageAfterScheduler(
    "amdgpu-print-max-reg-pressure-regusage-after-scheduler", cl::Hidden,
    cl::desc("Print a list of live registers along with their def/uses at the "
             "point of maximum register pressure after scheduling."),
    cl::init(false));
#endif
static cl::opt<bool>
    RematLiveThru("amdgpu-remat-livethru", cl::Hidden,
                  cl::desc("Rematerialize the LiveThru registers for the first "
                           "loop found in the code"),
                  cl::init(true));

static cl::opt<bool> RematLiveIn(
    "amdgpu-remat-into", cl::Hidden,
    cl::desc("Rematerialize any LiveIn registers for the first loop found in "
             "the code (may rematerialize into body of loop)"),
    cl::init(false));

static cl::opt<bool> DisableRewriteMFMAFormSchedStage(
    "amdgpu-disable-rewrite-mfma-form-sched-stage", cl::Hidden,
    cl::desc("Disable rewrie mfma rewrite scheduling stage"), cl::init(true));
static cl::opt<bool> DisableRemat(
    "amdgpu-disable-remat", cl::Hidden,
    cl::desc("Disable rematerialization during AMDGPU scheduling)"),
    cl::init(false));


const unsigned ScheduleMetrics::ScaleFactor = 100;

GCNSchedStrategy::GCNSchedStrategy(const MachineSchedContext *C)
    : GenericScheduler(C), TargetOccupancy(0), MF(nullptr),
      DownwardTracker(*C->LIS), UpwardTracker(*C->LIS), HasHighPressure(false) {
}

void GCNSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  GenericScheduler::initialize(DAG);

  MF = &DAG->MF;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();

  SGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::SGPR_32RegClass);

  VGPRExcessLimit =
      Context->RegClassInfo->getNumAllocatableRegs(&AMDGPU::VGPR_32RegClass);

  SIMachineFunctionInfo &MFI = *MF->getInfo<SIMachineFunctionInfo>();
  // Set the initial TargetOccupnacy to the maximum occupancy that we can
  // achieve for this function. This effectively sets a lower bound on the
  // 'Critical' register limits in the scheduler.
  // Allow for lower occupancy targets if kernel is wave limited or memory
  // bound, and using the relaxed occupancy feature.
  TargetOccupancy =
      RelaxedOcc ? MFI.getMinAllowedOccupancy() : MFI.getOccupancy();
  SGPRCriticalLimit =
      std::min(ST.getMaxNumSGPRs(TargetOccupancy, true), SGPRExcessLimit);

  if (!KnownExcessRP) {
    VGPRCriticalLimit = std::min(
        ST.getMaxNumVGPRs(TargetOccupancy, MFI.getDynamicVGPRBlockSize()),
        VGPRExcessLimit);
  } else {
    // This is similar to ST.getMaxNumVGPRs(TargetOccupancy) result except
    // returns a reasonably small number for targets with lots of VGPRs, such
    // as GFX10 and GFX11.
    LLVM_DEBUG(dbgs() << "Region is known to spill, use alternative "
                         "VGPRCriticalLimit calculation method.\n");
    unsigned DynamicVGPRBlockSize = MFI.getDynamicVGPRBlockSize();
    unsigned Granule =
        AMDGPU::IsaInfo::getVGPRAllocGranule(&ST, DynamicVGPRBlockSize);
    unsigned Addressable =
        AMDGPU::IsaInfo::getAddressableNumVGPRs(&ST, DynamicVGPRBlockSize);
    unsigned VGPRBudget = alignDown(Addressable / TargetOccupancy, Granule);
    VGPRBudget = std::max(VGPRBudget, Granule);
    VGPRCriticalLimit = std::min(VGPRBudget, VGPRExcessLimit);
  }

  // Subtract error margin and bias from register limits and avoid overflow.
  SGPRCriticalLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRCriticalLimit);
  VGPRCriticalLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRCriticalLimit);
  SGPRExcessLimit -= std::min(SGPRLimitBias + ErrorMargin, SGPRExcessLimit);
  VGPRExcessLimit -= std::min(VGPRLimitBias + ErrorMargin, VGPRExcessLimit);

  LLVM_DEBUG(dbgs() << "VGPRCriticalLimit = " << VGPRCriticalLimit
                    << ", VGPRExcessLimit = " << VGPRExcessLimit
                    << ", SGPRCriticalLimit = " << SGPRCriticalLimit
                    << ", SGPRExcessLimit = " << SGPRExcessLimit << "\n\n");
}

/// Checks whether \p SU can use the cached DAG pressure diffs to compute the
/// current register pressure.
///
/// This works for the common case, but it has a few exceptions that have been
/// observed through trial and error:
///   - Explicit physical register operands
///   - Subregister definitions
///
/// In both of those cases, PressureDiff doesn't represent the actual pressure,
/// and querying LiveIntervals through the RegPressureTracker is needed to get
/// an accurate value.
///
/// We should eventually only use PressureDiff for maximum performance, but this
/// already allows 80% of SUs to take the fast path without changing scheduling
/// at all. Further changes would either change scheduling, or require a lot
/// more logic to recover an accurate pressure estimate from the PressureDiffs.
static bool canUsePressureDiffs(const SUnit &SU) {
  if (!SU.isInstr())
    return false;

  // Cannot use pressure diffs for subregister defs or with physregs, it's
  // imprecise in both cases.
  for (const auto &Op : SU.getInstr()->operands()) {
    if (!Op.isReg() || Op.isImplicit())
      continue;
    if (Op.getReg().isPhysical() ||
        (Op.isDef() && Op.getSubReg() != AMDGPU::NoSubRegister))
      return false;
  }
  return true;
}

static void getRegisterPressures(
    bool AtTop, const RegPressureTracker &RPTracker, SUnit *SU,
    std::vector<unsigned> &Pressure, std::vector<unsigned> &MaxPressure,
    GCNDownwardRPTracker &DownwardTracker, GCNUpwardRPTracker &UpwardTracker,
    ScheduleDAGMI *DAG, const SIRegisterInfo *SRI) {
  // getDownwardPressure() and getUpwardPressure() make temporary changes to
  // the tracker, so we need to pass those function a non-const copy.
  RegPressureTracker &TempTracker = const_cast<RegPressureTracker &>(RPTracker);
  if (!GCNTrackers) {
    AtTop
        ? TempTracker.getDownwardPressure(SU->getInstr(), Pressure, MaxPressure)
        : TempTracker.getUpwardPressure(SU->getInstr(), Pressure, MaxPressure);

    return;
  }

  // GCNTrackers
  Pressure.resize(4, 0);
  MachineInstr *MI = SU->getInstr();
  GCNRegPressure NewPressure;
  if (AtTop) {
    GCNDownwardRPTracker TempDownwardTracker(DownwardTracker);
    NewPressure = TempDownwardTracker.bumpDownwardPressure(MI, SRI);
  } else {
    GCNUpwardRPTracker TempUpwardTracker(UpwardTracker);
    TempUpwardTracker.recede(*MI);
    NewPressure = TempUpwardTracker.getPressure();
  }
  Pressure[AMDGPU::RegisterPressureSets::SReg_32] = NewPressure.getSGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::VGPR_32] =
      NewPressure.getArchVGPRNum();
  Pressure[AMDGPU::RegisterPressureSets::AGPR_32] = NewPressure.getAGPRNum();
}

void GCNSchedStrategy::initCandidate(SchedCandidate &Cand, SUnit *SU,
                                     bool AtTop,
                                     const RegPressureTracker &RPTracker,
                                     const SIRegisterInfo *SRI,
                                     unsigned SGPRPressure,
                                     unsigned VGPRPressure, bool IsBottomUp) {
  Cand.SU = SU;
  Cand.AtTop = AtTop;

  if (!DAG->isTrackingPressure())
    return;

  Pressure.clear();
  MaxPressure.clear();

  // We try to use the cached PressureDiffs in the ScheduleDAG whenever
  // possible over querying the RegPressureTracker.
  //
  // RegPressureTracker will make a lot of LIS queries which are very
  // expensive, it is considered a slow function in this context.
  //
  // PressureDiffs are precomputed and cached, and getPressureDiff is just a
  // trivial lookup into an array. It is pretty much free.
  //
  // In EXPENSIVE_CHECKS, we always query RPTracker to verify the results of
  // PressureDiffs.
  if (AtTop || !canUsePressureDiffs(*SU) || GCNTrackers) {
    getRegisterPressures(AtTop, RPTracker, SU, Pressure, MaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
  } else {
    // Reserve 4 slots.
    Pressure.resize(4, 0);
    Pressure[AMDGPU::RegisterPressureSets::SReg_32] = SGPRPressure;
    Pressure[AMDGPU::RegisterPressureSets::VGPR_32] = VGPRPressure;

    for (const auto &Diff : DAG->getPressureDiff(SU)) {
      if (!Diff.isValid())
        continue;
      // PressureDiffs is always bottom-up so if we're working top-down we need
      // to invert its sign.
      Pressure[Diff.getPSet()] +=
          (IsBottomUp ? Diff.getUnitInc() : -Diff.getUnitInc());
    }

#ifdef EXPENSIVE_CHECKS
    std::vector<unsigned> CheckPressure, CheckMaxPressure;
    getRegisterPressures(AtTop, RPTracker, SU, CheckPressure, CheckMaxPressure,
                         DownwardTracker, UpwardTracker, DAG, SRI);
    if (Pressure[AMDGPU::RegisterPressureSets::SReg_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] ||
        Pressure[AMDGPU::RegisterPressureSets::VGPR_32] !=
            CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32]) {
       errs() << "Register Pressure is inaccurate when calculated through "
      "PressureDiff\n"
          << "SGPR got " << Pressure[AMDGPU::RegisterPressureSets::SReg_32]
          << ", expected "
          << CheckPressure[AMDGPU::RegisterPressureSets::SReg_32] << "\n"
          << "VGPR got " << Pressure[AMDGPU::RegisterPressureSets::VGPR_32]
          << ", expected "
          << CheckPressure[AMDGPU::RegisterPressureSets::VGPR_32] << "\n";
      report_fatal_error("inaccurate register pressure calculation");
    }
#endif
  }

  unsigned NewSGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
  unsigned NewVGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];

  // If two instructions increase the pressure of different register sets
  // by the same amount, the generic scheduler will prefer to schedule the
  // instruction that increases the set with the least amount of registers,
  // which in our case would be SGPRs.  This is rarely what we want, so
  // when we report excess/critical register pressure, we do it either
  // only for VGPRs or only for SGPRs.

  // FIXME: Better heuristics to determine whether to prefer SGPRs or VGPRs.
  const unsigned MaxVGPRPressureInc = 16;
  bool ShouldTrackSGPRs = SGPRPressure + MaxVGPRPressureInc >= SGPRExcessLimit;
  bool ShouldTrackVGPRs = !ShouldTrackSGPRs && VGPRPressure >= VGPRExcessLimit;

  // FIXME: We have to enter REG-EXCESS before we reach the actual threshold
  // to increase the likelihood we don't go over the limits.  We should improve
  // the analysis to look through dependencies to find the path with the least
  // register pressure.

  // We only need to update the RPDelta for instructions that increase register
  // pressure. Instructions that decrease or keep reg pressure the same will be
  // marked as RegExcess in tryCandidate() when they are compared with
  // instructions that increase the register pressure.
  if (ShouldTrackVGPRs && NewVGPRPressure >= VGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
    Cand.RPDelta.Excess.setUnitInc(NewVGPRPressure - VGPRExcessLimit);
  }

  if (ShouldTrackSGPRs && NewSGPRPressure >= SGPRExcessLimit) {
    HasHighPressure = true;
    Cand.RPDelta.Excess = PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
    Cand.RPDelta.Excess.setUnitInc(NewSGPRPressure - SGPRExcessLimit);
  }

  // Register pressure is considered 'CRITICAL' if it is approaching a value
  // that would reduce the wave occupancy for the execution unit.  When
  // register pressure is 'CRITICAL', increasing SGPR and VGPR pressure both
  // has the same cost, so we don't need to prefer one over the other.

  int SGPRDelta = NewSGPRPressure - SGPRCriticalLimit;
  int VGPRDelta = NewVGPRPressure - VGPRCriticalLimit;

  if (VGPRDelta >= 0 || SGPRDelta >= 0) {
    HasHighPressure = true;
    if (VGPRDelta > SGPRDelta) {
      Cand.RPDelta.CriticalMax =
        PressureChange(AMDGPU::RegisterPressureSets::VGPR_32);
      Cand.RPDelta.CriticalMax.setUnitInc(VGPRDelta);
    } else {
      Cand.RPDelta.CriticalMax =
          PressureChange(AMDGPU::RegisterPressureSets::SReg_32);
      Cand.RPDelta.CriticalMax.setUnitInc(SGPRDelta);
    }
  }
}

static bool shouldCheckPending(SchedBoundary &Zone,
                               const TargetSchedModel *SchedModel) {
  bool HasBufferedModel =
      SchedModel->hasInstrSchedModel() && SchedModel->getMicroOpBufferSize();
  unsigned Combined = Zone.Available.size() + Zone.Pending.size();
  return Combined <= PendingQueueLimit && HasBufferedModel;
}

static SUnit *pickOnlyChoice(SchedBoundary &Zone,
                             const TargetSchedModel *SchedModel) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!shouldCheckPending(Zone, SchedModel) || Zone.Pending.empty())
    return OnlyChoice;

  return nullptr;
}

void GCNSchedStrategy::printCandidateDecision(const SchedCandidate &Current,
                                              const SchedCandidate &Preferred) {
  LLVM_DEBUG({
    dbgs() << "Prefer:\t\t";
    DAG->dumpNode(*Preferred.SU);

    if (Current.SU) {
      dbgs() << "Not:\t";
      DAG->dumpNode(*Current.SU);
    }

    dbgs() << "Reason:\t\t";
    traceCandidate(Preferred);
  });
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeFromQueue()
void GCNSchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
                                         const RegPressureTracker &RPTracker,
                                         SchedCandidate &Cand, bool &IsPending,
                                         bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo*>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  IsPending = false;
  if (DAG->isTrackingPressure()) {
    if (!GCNTrackers) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      VGPRPressure = T->getPressure().getArchVGPRNum();
    }
  }
  LLVM_DEBUG(dbgs() << "Available Q:\n");
  ReadyQueue &AQ = Zone.Available;
  for (SUnit *SU : AQ) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }

  if (!shouldCheckPending(Zone, SchedModel))
    return;

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  ReadyQueue &PQ = Zone.Pending;
  for (SUnit *SU : PQ) {

    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryPendingCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      LLVM_DEBUG(printCandidateDecision(Cand, TryCand));
      IsPending = true;
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNodeBidirectional()
SUnit *GCNSchedStrategy::pickNodeBidirectional(bool &IsTopNode,
                                               bool &PickedPending) {
  // Schedule as far as possible in the direction of no choice. This is most
  // efficient, but also provides the best heuristics for CriticalPSets.
  if (SUnit *SU = pickOnlyChoice(Bot, SchedModel)) {
    IsTopNode = false;
    return SU;
  }
  if (SUnit *SU = pickOnlyChoice(Top, SchedModel)) {
    IsTopNode = true;
    return SU;
  }
  // Set the bottom-up policy based on the state of the current bottom zone
  // and the instructions outside the zone, including the top zone.
  CandPolicy BotPolicy;
  setPolicy(BotPolicy, /*IsPostRA=*/false, Bot, &Top);
  // Set the top-down policy based on the state of the current top zone and
  // the instructions outside the zone, including the bottom zone.
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, /*IsPostRA=*/false, Top, &Bot);

  bool BotPending = false;
  // See if BotCand is still valid (because we previously scheduled from Top).
  LLVM_DEBUG(dbgs() << "Picking from Bot:\n");
  if (!BotCand.isValid() || BotCand.SU->isScheduled ||
      BotCand.Policy != BotPolicy) {
    BotCand.reset(CandPolicy());
    pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), BotCand,
                      BotPending,
                      /*IsBottomUp=*/true);
    assert(BotCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(BotCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Bot, BotPolicy, DAG->getBotRPTracker(), TCand,
                        BotPending,
                        /*IsBottomUp=*/true);
      assert(TCand.SU == BotCand.SU &&
             "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  bool TopPending = false;
  // Check if the top Q has a better candidate.
  LLVM_DEBUG(dbgs() << "Picking from Top:\n");
  if (!TopCand.isValid() || TopCand.SU->isScheduled ||
      TopCand.Policy != TopPolicy) {
    TopCand.reset(CandPolicy());
    pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TopCand,
                      TopPending,
                      /*IsBottomUp=*/false);
    assert(TopCand.Reason != NoCand && "failed to find the first candidate");
  } else {
    LLVM_DEBUG(traceCandidate(TopCand));
#ifndef NDEBUG
    if (VerifyScheduling) {
      SchedCandidate TCand;
      TCand.reset(CandPolicy());
      pickNodeFromQueue(Top, TopPolicy, DAG->getTopRPTracker(), TCand,
                        TopPending,
                        /*IsBottomUp=*/false);
      assert(TCand.SU == TopCand.SU &&
           "Last pick result should correspond to re-picking right now");
    }
#endif
  }

  // Pick best from BotCand and TopCand.
  LLVM_DEBUG(dbgs() << "Top Cand: "; traceCandidate(TopCand);
             dbgs() << "Bot Cand: "; traceCandidate(BotCand););
  SchedCandidate Cand = BotPending ? TopCand : BotCand;
  SchedCandidate TryCand = BotPending ? BotCand : TopCand;
  PickedPending = BotPending && TopPending;

  TryCand.Reason = NoCand;
  if (BotPending || TopPending) {
    PickedPending |= tryPendingCandidate(Cand, TopCand, nullptr);
  } else {
    tryCandidate(Cand, TryCand, nullptr);
  }

  if (TryCand.Reason != NoCand) {
    Cand.setBest(TryCand);
  }

  LLVM_DEBUG(dbgs() << "Picking: "; traceCandidate(Cand););

  IsTopNode = Cand.AtTop;
  return Cand.SU;
}

// This function is mostly cut and pasted from
// GenericScheduler::pickNode()
SUnit *GCNSchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  bool PickedPending;
  if (CustomResTracking)
    RegionPolicy.OnlyTopDown = true;
  SUnit *SU;
  do {
    PickedPending = false;
    if (RegionPolicy.OnlyTopDown) {
      SU = pickOnlyChoice(Top, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                          PickedPending,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = pickOnlyChoice(Bot, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand,
                          PickedPending,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode, PickedPending);
    }
  } while (SU->isScheduled);

  if (PickedPending) {
    unsigned ReadyCycle = IsTopNode ? SU->TopReadyCycle : SU->BotReadyCycle;
    SchedBoundary &Zone = IsTopNode ? Top : Bot;
    unsigned CurrentCycle = Zone.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Zone.bumpCycle(ReadyCycle);

    // FIXME: checkHazard() doesn't give information about which cycle the
    // hazard will resolve so just keep bumping the cycle by 1. This could be
    // made more efficient if checkHazard() returned more details.
    while (Zone.checkHazard(SU))
      Zone.bumpCycle(Zone.getCurrCycle() + 1);

    Zone.releasePending();
  }

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  if (CustomResTracking) {
#ifndef NDEBUG
    unsigned XDLCyclesBefore = XDLProcRes.CyclesReserved;
#endif

    const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(DAG->TII);
    bool IsXDL = TII->isXDL(*SU->getInstr());
    unsigned Cycles = SU->Latency;
    if (IsXDL) {
      // FIXME: Hack since XDL is only actually occupying for 24 cycles with 8
      // pass MFMA.
      if (Cycles > 2)
        Cycles -= 2;
      XDLProcRes.reset();
      XDLProcRes.reserve(Cycles);
    } else {
      XDLProcRes.release(Cycles);
    }

    LLVM_DEBUG(dbgs() << "OldXDLProcRes: " << XDLCyclesBefore
                      << "\nNewXDLProcRes: " << XDLProcRes.CyclesReserved
                      << "\n");
  }

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

void GCNSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  if (GCNTrackers) {
    MachineInstr *MI = SU->getInstr();
    IsTopNode ? (void)DownwardTracker.advance(MI, false)
              : UpwardTracker.recede(*MI);
  }

  return GenericScheduler::schedNode(SU, IsTopNode);
}

bool GCNMaxOccupancySchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                SchedCandidate &TryCand,
                                                SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (CustomResTracking && tryXDL(Cand, TryCand, Zone))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
    // Keep clustered nodes together.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  if (SameBoundary) {
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

GCNSchedStageID GCNSchedStrategy::getCurrentStage() {
  assert(CurrentStage && CurrentStage != SchedStages.end());
  return *CurrentStage;
}

bool GCNSchedStrategy::advanceStage() {
  assert(CurrentStage != SchedStages.end());
  if (!CurrentStage)
    CurrentStage = SchedStages.begin();
  else
    CurrentStage++;

  return CurrentStage != SchedStages.end();
}

bool GCNSchedStrategy::hasNextStage() const {
  assert(CurrentStage);
  return std::next(CurrentStage) != SchedStages.end();
}

GCNSchedStageID GCNSchedStrategy::getNextStage() const {
  assert(CurrentStage && std::next(CurrentStage) != SchedStages.end());
  return *std::next(CurrentStage);
}

bool GCNSchedStrategy::tryPendingCandidate(SchedCandidate &Cand,
                                           SchedCandidate &TryCand,
                                           SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;
  }

  return false;
}

GCNMaxOccupancySchedStrategy::GCNMaxOccupancySchedStrategy(
    const MachineSchedContext *C, bool IsLegacyScheduler)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::OccInitialSchedule);
  if (!DisableRewriteMFMAFormSchedStage)
    SchedStages.push_back(GCNSchedStageID::RewriteMFMAForm);
  SchedStages.push_back(GCNSchedStageID::UnclusteredHighRPReschedule);
  SchedStages.push_back(GCNSchedStageID::ClusteredLowOccupancyReschedule);
  if (!DisableRemat) SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  GCNTrackers = GCNTrackers & !IsLegacyScheduler;
}

GCNMaxILPSchedStrategy::GCNMaxILPSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
}

bool GCNMaxILPSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                          SchedCandidate &TryCand,
                                          SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Avoid spilling by exceeding the register limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;

    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Unconditionally try to reduce latency.
    if (tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
      TryCand.Reason = NodeOrder;
      return true;
    }
  }
  return false;
}

GCNMaxMemoryClauseSchedStrategy::GCNMaxMemoryClauseSchedStrategy(
    const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::MemoryClauseInitialSchedule);
}

bool GCNSchedStrategy::tryXDL(SchedCandidate &Cand, SchedCandidate &TryCand,
                              SchedBoundary *Zone) const {
  assert(Zone->isTop());
  MachineInstr *CInst = Cand.SU->getInstr();
  MachineInstr *TCInst = TryCand.SU->getInstr();
  const SIInstrInfo *TII = DAG->MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  bool CandIsXDL = TII->isXDL(*CInst);
  bool TryCandIsXDL = TII->isXDL(*TCInst);

  LLVM_DEBUG(dbgs() << "tryXDL"; DAG->dumpNode(*Cand.SU);
             DAG->dumpNode(*TryCand.SU));

  // XDL is free.
  if (XDLProcRes.CyclesReserved == 0) {
    if (!TryCandIsXDL && !CandIsXDL)
      return false;

    if (TryCandIsXDL && !CandIsXDL) {
      TryCand.Reason = ResourceDemand;
      return true;
    }

    if (CandIsXDL && !TryCandIsXDL) {
      Cand.Reason = ResourceDemand;
      return true;
    }

    // XDL is free and both candidates are XDL.
    if (CandIsXDL && TryCandIsXDL) {
      unsigned CandReadyVALUSuccs = 0;
      unsigned TryReadyVALUSuccs = 0;

      // Count VALU successors that would become ready after scheduling this
      // MFMA.
      SmallPtrSet<SUnit *, 8> CandSeenSuccs;
      for (SDep &Succ : Cand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!CandSeenSuccs.insert(SuccSU).second)
          continue;

        if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
          ++CandReadyVALUSuccs;
        }
      }

      SmallPtrSet<SUnit *, 8> TrySeenSuccs;
      for (SDep &Succ : TryCand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!TrySeenSuccs.insert(SuccSU).second)
          continue;

        if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
          ++TryReadyVALUSuccs;
        }
      }

      LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                        << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

      // Prefer the candidate that would immediately free up more VALU
      // instructions
      if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
        Cand.Reason = ResourceDemand;
        return true;
      }
      if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
        TryCand.Reason = ResourceDemand;
        return true;
      }
      if (CandReadyVALUSuccs > 0) {
        Cand.Reason = ResourceDemand;
        return true;
      }

      // If they free up the same number of VALUs, fall back to the old
      // heuristic and just count the total number of VALU successors
      unsigned CandVALUSuccs = 0;
      unsigned TryVALUSuccs = 0;

      // Reset and reuse our sets of seen successors
      CandSeenSuccs.clear();
      TrySeenSuccs.clear();

      for (SDep &Succ : Cand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!CandSeenSuccs.insert(SuccSU).second)
          continue;
        if (TII->isVALU(*SuccSU->getInstr()))
          ++CandVALUSuccs;
      }

      for (SDep &Succ : TryCand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!TrySeenSuccs.insert(SuccSU).second)
          continue;
        if (TII->isVALU(*SuccSU->getInstr()))
          ++TryVALUSuccs;
      }

      LLVM_DEBUG(dbgs() << "CandVALUSuccs: " << CandVALUSuccs
                        << " TryVALUSuccs: " << TryVALUSuccs << "\n");

      // If one candidate has more total VALU successors, prefer it.
      if (CandVALUSuccs > TryVALUSuccs) {
        Cand.Reason = ResourceDemand;
        return true;
      }
      if (CandVALUSuccs < TryVALUSuccs) {
        TryCand.Reason = ResourceDemand;
        return true;
      }

      Cand.Reason = ResourceDemand;
      return true;
    }
  }

  assert(XDLProcRes.CyclesReserved);

  // XDL is in use and Cand is a MFMA.
  if (CandIsXDL) {
    if (!TryCandIsXDL) {
      TryCand.Reason = ResourceReduce;
      return true;
    }
    // TryCandIsXDL and CandIsXDL and resource is in use.
    unsigned CandReadyVALUSuccs = 0;
    unsigned TryReadyVALUSuccs = 0;

    // Count VALU successors that would become ready after scheduling this
    // MFMA.
    SmallPtrSet<SUnit *, 8> CandSeenSuccs;
    for (SDep &Succ : Cand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!CandSeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++CandReadyVALUSuccs;
      }
    }

    SmallPtrSet<SUnit *, 8> TrySeenSuccs;
    for (SDep &Succ : TryCand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!TrySeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++TryReadyVALUSuccs;
      }
    }

    LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                      << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

    // Prefer the candidate that would immediately free up more VALU
    // instructions
    if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
      TryCand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs > 0) {
      Cand.Reason = ResourceDemand;
      return true;
    }

    // Let other heuristics take precedence if both are MFMA and resource is
    // in use.
    return false;
  }

  // Both candidates are not MFMA and resource is in use.
  if (!TryCandIsXDL) {
    unsigned CandReadyVALUSuccs = 0;
    unsigned TryReadyVALUSuccs = 0;

    SmallPtrSet<SUnit *, 8> CandSeenSuccs;
    for (SDep &Succ : Cand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!CandSeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++CandReadyVALUSuccs;
      }
    }

    SmallPtrSet<SUnit *, 8> TrySeenSuccs;
    for (SDep &Succ : TryCand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!TrySeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++TryReadyVALUSuccs;
      }
    }

    LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                      << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

    // Prefer the candidate that would immediately free up more VALU
    // instructions
    if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
      TryCand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs > 0) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    unsigned CandCycles = Cand.SU->Latency;
    unsigned TryCandCycles = TryCand.SU->Latency;

    // Check if either instruction would cause XDL resources to go negative
    bool CandOverflow = CandCycles > XDLProcRes.CyclesReserved;
    bool TryCandOverflow = TryCandCycles > XDLProcRes.CyclesReserved;

    // If one overflows and the other doesn't, prefer the one that overflows
    // because it will free up the XDL resource
    if (CandOverflow && !TryCandOverflow) {
      Cand.Reason = ResourceReduce;
      return true;
    }
    if (TryCandOverflow && !CandOverflow) {
      TryCand.Reason = ResourceReduce;
      return true;
    }

    // Both would overflow or neither would - pick the one that gets closest to
    // zero
    int CandRemainingXDL = static_cast<int>(XDLProcRes.CyclesReserved) -
                           static_cast<int>(CandCycles);
    int TryCandRemainingXDL = static_cast<int>(XDLProcRes.CyclesReserved) -
                              static_cast<int>(TryCandCycles);

    if (std::abs(TryCandRemainingXDL) < std::abs(CandRemainingXDL)) {
      TryCand.Reason = ResourceReduce;
      return true;
    }
    Cand.Reason = ResourceReduce;
    return true;
  }

  // XDL resource is in use and Cand is not MFMA but TryCand is.
  Cand.Reason = ResourceReduce;
  return true;
}

/// GCNMaxMemoryClauseSchedStrategy tries best to clause memory instructions as
/// much as possible. This is achieved by:
//  1. Prioritize clustered operations before stall latency heuristic.
//  2. Prioritize long-latency-load before stall latency heuristic.
///
/// \param Cand provides the policy and current best candidate.
/// \param TryCand refers to the next SUnit candidate, otherwise uninitialized.
/// \param Zone describes the scheduled zone that we are extending, or nullptr
///             if Cand is from a different zone than TryCand.
/// \return \c true if TryCand is better than Cand (Reason is NOT NoCand)
bool GCNMaxMemoryClauseSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                                   SchedCandidate &TryCand,
                                                   SchedBoundary *Zone) const {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg))
    return TryCand.Reason != NoCand;

  if (DAG->isTrackingPressure()) {
    // Avoid exceeding the target's limit.
    if (tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                    RegExcess, TRI, DAG->MF))
      return TryCand.Reason != NoCand;

    // Avoid increasing the max critical pressure in the scheduled region.
    if (tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                    TryCand, Cand, RegCritical, TRI, DAG->MF))
      return TryCand.Reason != NoCand;
  }

  // MaxMemoryClause-specific: We prioritize clustered instructions as we would
  // get more benefit from clausing these memory instructions.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);
  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // MaxMemoryClause-specific: Prioritize long latency memory load
    // instructions in top-bottom order to hide more latency. The mayLoad check
    // is used to exclude store-like instructions, which we do not want to
    // scheduler them too early.
    bool TryMayLoad =
        TryCand.SU->isInstr() && TryCand.SU->getInstr()->mayLoad();
    bool CandMayLoad = Cand.SU->isInstr() && Cand.SU->getInstr()->mayLoad();

    if (TryMayLoad || CandMayLoad) {
      bool TryLongLatency =
          TryCand.SU->Latency > 10 * Cand.SU->Latency && TryMayLoad;
      bool CandLongLatency =
          10 * TryCand.SU->Latency < Cand.SU->Latency && CandMayLoad;

      if (tryGreater(Zone->isTop() ? TryLongLatency : CandLongLatency,
                     Zone->isTop() ? CandLongLatency : TryLongLatency, TryCand,
                     Cand, Stall))
        return TryCand.Reason != NoCand;
    }
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Zone->getLatencyStallCycles(TryCand.SU),
                Zone->getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
      return TryCand.Reason != NoCand;
  }

  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak))
      return TryCand.Reason != NoCand;
  }

  // Avoid increasing the max pressure of the entire region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
                  Cand, RegMax, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  if (SameBoundary) {
    // Avoid critical resource consumption and balance the schedule.
    TryCand.initResourceDelta(DAG, SchedModel);
    if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
                TryCand, Cand, ResourceReduce))
      return TryCand.Reason != NoCand;
    if (tryGreater(TryCand.ResDelta.DemandedResources,
                   Cand.ResDelta.DemandedResources, TryCand, Cand,
                   ResourceDemand))
      return TryCand.Reason != NoCand;

    // Avoid serializing long latency dependence chains.
    // For acyclic path limited loops, latency was already checked above.
    if (!RegionPolicy.DisableLatencyHeuristic && TryCand.Policy.ReduceLatency &&
        !Rem.IsAcyclicLatencyLimited && tryLatency(TryCand, Cand, *Zone))
      return TryCand.Reason != NoCand;

    // Fall through to original instruction order.
    if (Zone->isTop() == (TryCand.SU->NodeNum < Cand.SU->NodeNum)) {
      assert(TryCand.SU->NodeNum != Cand.SU->NodeNum);
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

GCNScheduleDAGMILive::GCNScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGMILive(C, std::move(S)), ST(MF.getSubtarget<GCNSubtarget>()),
      MFI(*MF.getInfo<SIMachineFunctionInfo>()),
      StartingOccupancy(MFI.getOccupancy()), MinOccupancy(StartingOccupancy),
      RegionLiveOuts(this, /*IsLiveOut=*/true) {

  ScheduleSingleMIRegions = false;
  LLVM_DEBUG(dbgs() << "Starting occupancy is " << StartingOccupancy << ".\n");
  if (RelaxedOcc) {
    MinOccupancy = std::min(MFI.getMinAllowedOccupancy(), StartingOccupancy);
    if (MinOccupancy != StartingOccupancy)
      LLVM_DEBUG(dbgs() << "Allowing Occupancy drops to " << MinOccupancy
                        << ".\n");
  }
}

std::unique_ptr<GCNSchedStage>
GCNScheduleDAGMILive::createSchedStage(GCNSchedStageID SchedStageID) {
  switch (SchedStageID) {
  case GCNSchedStageID::OccInitialSchedule:
    return std::make_unique<OccInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::RewriteMFMAForm:
    return std::make_unique<RewriteMFMAFormStage>(SchedStageID, *this);
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    return std::make_unique<UnclusteredHighRPStage>(SchedStageID, *this);
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    return std::make_unique<ClusteredLowOccStage>(SchedStageID, *this);
  case GCNSchedStageID::PreRARematerialize:
    return std::make_unique<PreRARematStage>(SchedStageID, *this);
  case GCNSchedStageID::ILPInitialSchedule:
    return std::make_unique<ILPInitialScheduleStage>(SchedStageID, *this);
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    return std::make_unique<MemoryClauseInitialScheduleStage>(SchedStageID,
                                                              *this);
  }

  llvm_unreachable("Unknown SchedStageID.");
}

void GCNScheduleDAGMILive::schedule() {
  // Collect all scheduling regions. The actual scheduling is performed in
  // GCNScheduleDAGMILive::finalizeSchedule.
  Regions.push_back(std::pair(RegionBegin, RegionEnd));
}

GCNRegPressure
GCNScheduleDAGMILive::getRealRegPressure(unsigned RegionIdx) const {
  GCNDownwardRPTracker RPTracker(*LIS);
  RPTracker.advance(Regions[RegionIdx].first, Regions[RegionIdx].second,
                    &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

static MachineInstr *getLastMIForRegion(MachineBasicBlock::iterator RegionBegin,
                                        MachineBasicBlock::iterator RegionEnd) {
  assert(RegionBegin != RegionEnd && "Region must not be empty");
  return &*skipDebugInstructionsBackward(std::prev(RegionEnd), RegionBegin);
}

void GCNScheduleDAGMILive::computeBlockPressure(unsigned RegionIdx,
                                                const MachineBasicBlock *MBB) {
  GCNDownwardRPTracker RPTracker(*LIS);

  // If the block has the only successor then live-ins of that successor are
  // live-outs of the current block. We can reuse calculated live set if the
  // successor will be sent to scheduling past current block.

  // However, due to the bug in LiveInterval analysis it may happen that two
  // predecessors of the same successor block have different lane bitmasks for
  // a live-out register. Workaround that by sticking to one-to-one relationship
  // i.e. one predecessor with one successor block.
  const MachineBasicBlock *OnlySucc = nullptr;
  if (MBB->succ_size() == 1) {
    auto *Candidate = *MBB->succ_begin();
    if (!Candidate->empty() && Candidate->pred_size() == 1) {
      SlotIndexes *Ind = LIS->getSlotIndexes();
      if (Ind->getMBBStartIdx(MBB) < Ind->getMBBStartIdx(Candidate))
        OnlySucc = Candidate;
    }
  }

  // Scheduler sends regions from the end of the block upwards.
  size_t CurRegion = RegionIdx;
  for (size_t E = Regions.size(); CurRegion != E; ++CurRegion)
    if (Regions[CurRegion].first->getParent() != MBB)
      break;
  --CurRegion;

  auto I = MBB->begin();
  auto LiveInIt = MBBLiveIns.find(MBB);
  auto &Rgn = Regions[CurRegion];
  auto *NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
  if (LiveInIt != MBBLiveIns.end()) {
    auto LiveIn = std::move(LiveInIt->second);
    RPTracker.reset(*MBB->begin(), &LiveIn);
    MBBLiveIns.erase(LiveInIt);
  } else {
    I = Rgn.first;
    auto LRS = BBLiveInMap.lookup(NonDbgMI);
#ifdef EXPENSIVE_CHECKS
    assert(isEqual(getLiveRegsBefore(*NonDbgMI, *LIS), LRS));
#endif
    RPTracker.reset(*I, &LRS);
  }

  for (;;) {
    I = RPTracker.getNext();

    if (Regions[CurRegion].first == I || NonDbgMI == I) {
      LiveIns[CurRegion] = RPTracker.getLiveRegs();
      RPTracker.clearMaxPressure();
    }

    if (Regions[CurRegion].second == I) {
      Pressure[CurRegion] = RPTracker.moveMaxPressure();
      if (CurRegion-- == RegionIdx)
        break;
      auto &Rgn = Regions[CurRegion];
      NonDbgMI = &*skipDebugInstructionsForward(Rgn.first, Rgn.second);
    }
    RPTracker.advanceToNext();
    RPTracker.advanceBeforeNext();
  }

  if (OnlySucc) {
    if (I != MBB->end()) {
      RPTracker.advanceToNext();
      RPTracker.advance(MBB->end());
    }
    RPTracker.advanceBeforeNext();
    MBBLiveIns[OnlySucc] = RPTracker.moveLiveRegs();
  }
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveInMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionFirstMIs;
  RegionFirstMIs.reserve(Regions.size());
  auto I = Regions.rbegin(), E = Regions.rend();
  do {
    auto *MI = &*skipDebugInstructionsForward(I->first, I->second);
    RegionFirstMIs.push_back(MI);
    ++I;
  } while (I != E);
  return getLiveRegMap(RegionFirstMIs, /*After=*/false, *LIS);
}

DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet>
GCNScheduleDAGMILive::getRegionLiveOutMap() const {
  assert(!Regions.empty());
  std::vector<MachineInstr *> RegionLastMIs;
  RegionLastMIs.reserve(Regions.size());
  for (auto &[RegionBegin, RegionEnd] : reverse(Regions)) {
    // Skip empty regions.
    if (RegionBegin == RegionEnd)
      continue;
    RegionLastMIs.push_back(getLastMIForRegion(RegionBegin, RegionEnd));
  }
  return getLiveRegMap(RegionLastMIs, /*After=*/true, *LIS);
}

void RegionPressureMap::buildLiveRegMap() {
  IdxToInstruction.clear();

  RegionLiveRegMap =
      IsLiveOut ? DAG->getRegionLiveOutMap() : DAG->getRegionLiveInMap();
  for (unsigned I = 0; I < DAG->Regions.size(); I++) {
    auto &[RegionBegin, RegionEnd] = DAG->Regions[I];
    // Skip empty regions.
    if (RegionBegin == RegionEnd)
      continue;
    MachineInstr *RegionKey =
        IsLiveOut ? getLastMIForRegion(RegionBegin, RegionEnd) : &*RegionBegin;
    IdxToInstruction[I] = RegionKey;
  }
}

void GCNScheduleDAGMILive::finalizeSchedule() {
  // Start actual scheduling here. This function is called by the base
  // MachineScheduler after all regions have been recorded by
  // GCNScheduleDAGMILive::schedule().
  LiveIns.resize(Regions.size());
  Pressure.resize(Regions.size());
  RegionsWithHighRP.resize(Regions.size());
  RegionsWithExcessRP.resize(Regions.size());
  RegionsWithIGLPInstrs.resize(Regions.size());
  RegionsWithHighRP.reset();
  RegionsWithExcessRP.reset();
  RegionsWithIGLPInstrs.reset();

  runSchedStages();
}

void GCNScheduleDAGMILive::runSchedStages() {
  LLVM_DEBUG(dbgs() << "All regions recorded, starting actual scheduling.\n");

  if (!Regions.empty()) {
    BBLiveInMap = getRegionLiveInMap();
    if (GCNTrackers)
      RegionLiveOuts.buildLiveRegMap();
  }

#ifdef DUMP_MAX_REG_PRESSURE
  if (PrintMaxRPRegUsageBeforeScheduler) {
    dumpMaxRegPressure(MF, GCNRegPressure::VGPR, *LIS, MLI);
    dumpMaxRegPressure(MF, GCNRegPressure::SGPR, *LIS, MLI);
    LIS->dump();
  }
#endif

  GCNSchedStrategy &S = static_cast<GCNSchedStrategy &>(*SchedImpl);
  while (S.advanceStage()) {
    auto Stage = createSchedStage(S.getCurrentStage());
    if (!Stage->initGCNSchedStage())
      continue;

    if (S.getCurrentStage() == GCNSchedStageID::PreRARematerialize) {
      BBLiveInMap = getRegionLiveInMap();
      RegionLiveOuts.buildLiveRegMap();
    }

    for (auto Region : Regions) {
      RegionBegin = Region.first;
      RegionEnd = Region.second;

      if (RegionBegin == RegionEnd)
        continue;

      // Setup for scheduling the region and check whether it should be skipped.
      if (!Stage->initGCNRegion()) {
        Stage->advanceRegion();
        exitRegion();
        continue;
      }

      if (GCNTrackers) {
        GCNDownwardRPTracker *DownwardTracker = S.getDownwardTracker();
        GCNUpwardRPTracker *UpwardTracker = S.getUpwardTracker();
        GCNRPTracker::LiveRegSet *RegionLiveIns =
            &LiveIns[Stage->getRegionIdx()];

        reinterpret_cast<GCNRPTracker *>(DownwardTracker)
            ->reset(MRI, *RegionLiveIns);
        reinterpret_cast<GCNRPTracker *>(UpwardTracker)
            ->reset(MRI, RegionLiveOuts.getLiveRegsForRegionIdx(
                             Stage->getRegionIdx()));


        GCNRegPressure LiveThru;
        for (auto LR : LiveIns[Stage->getRegionIdx()]) {
          bool FoundIt = false;
          for (auto &U : MRI.use_nodbg_instructions(LR.first)) {
            if (U.getParent() == RegionBegin->getParent()) {
             FoundIt = true;
             break;
            }

          }
          if (!FoundIt) {
            LiveThru.inc(LR.first, (LaneBitmask)0, LR.second, MRI);
          }
        }
      }

      ScheduleDAGMILive::schedule();
      Stage->finalizeGCNRegion();
    }

    Stage->finalizeGCNSchedStage();
  }

#ifdef DUMP_MAX_REG_PRESSURE
  if (PrintMaxRPRegUsageAfterScheduler) {
    dumpMaxRegPressure(MF, GCNRegPressure::VGPR, *LIS, MLI);
    dumpMaxRegPressure(MF, GCNRegPressure::SGPR, *LIS, MLI);
    LIS->dump();
  }
#endif
}

#ifndef NDEBUG
raw_ostream &llvm::operator<<(raw_ostream &OS, const GCNSchedStageID &StageID) {
  switch (StageID) {
  case GCNSchedStageID::OccInitialSchedule:
    OS << "Max Occupancy Initial Schedule";
    break;
  case GCNSchedStageID::RewriteMFMAForm:
    OS << "Instruction Rewriting Reschedule";
    break;
  case GCNSchedStageID::UnclusteredHighRPReschedule:
    OS << "Unclustered High Register Pressure Reschedule";
    break;
  case GCNSchedStageID::ClusteredLowOccupancyReschedule:
    OS << "Clustered Low Occupancy Reschedule";
    break;
  case GCNSchedStageID::PreRARematerialize:
    OS << "Pre-RA Rematerialize";
    break;
  case GCNSchedStageID::ILPInitialSchedule:
    OS << "Max ILP Initial Schedule";
    break;
  case GCNSchedStageID::MemoryClauseInitialSchedule:
    OS << "Max memory clause Initial Schedule";
    break;
  }

  return OS;
}
#endif

GCNSchedStage::GCNSchedStage(GCNSchedStageID StageID, GCNScheduleDAGMILive &DAG)
    : DAG(DAG), S(static_cast<GCNSchedStrategy &>(*DAG.SchedImpl)), MF(DAG.MF),
      MFI(DAG.MFI), ST(DAG.ST), StageID(StageID) {}

bool GCNSchedStage::initGCNSchedStage() {
  if (!DAG.LIS)
    return false;

  LLVM_DEBUG(dbgs() << "Starting scheduling stage: " << StageID << "\n");
  return true;
}

void RewriteMFMAFormStage::findReachingDefs(
    MachineOperand &UseMO, LiveIntervals *LIS,
    SmallVectorImpl<SlotIndex> &DefIdxs) {
  MachineInstr *UseMI = UseMO.getParent();
  LiveInterval &UseLI = LIS->getInterval(UseMO.getReg());
  VNInfo *VNI = UseLI.getVNInfoAt(LIS->getInstructionIndex(*UseMI));

  // If the def is not a PHI, then it must be the only reaching def.
  if (!VNI->isPHIDef()) {
    DefIdxs.push_back(VNI->def);
    return;
  }

  SmallPtrSet<MachineBasicBlock *, 8> Visited = {UseMI->getParent()};
  SmallVector<MachineBasicBlock *, 8> Worklist;

  // Mark the predecessor blocks for traversal
  for (MachineBasicBlock *PredMBB : UseMI->getParent()->predecessors()) {
    Worklist.push_back(PredMBB);
    Visited.insert(PredMBB);
  }

  while (!Worklist.empty()) {
    MachineBasicBlock *CurrMBB = Worklist.pop_back_val();

    SlotIndex CurrMBBEnd = LIS->getMBBEndIdx(CurrMBB);
    VNInfo *VNI = UseLI.getVNInfoAt(CurrMBBEnd.getPrevSlot());

    MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);

    // If there is a def in this block, then add it to the list. This is the
    // reaching def of this path.
    if (!VNI->isPHIDef()) {
      DefIdxs.push_back(VNI->def);
      continue;
    }

    for (MachineBasicBlock *PredMBB : DefMBB->predecessors()) {
      if (Visited.insert(PredMBB).second)
        Worklist.push_back(PredMBB);
    }
  }
}

void RewriteMFMAFormStage::findReachingUses(
    MachineInstr *DefMI, LiveIntervals *LIS,
    SmallVectorImpl<MachineOperand *> &ReachingUses) {
  SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
  for (MachineOperand &UseMO :
       DAG.MRI.use_nodbg_operands(DefMI->getOperand(0).getReg())) {
    SmallVector<SlotIndex, 8> ReachingDefIndexes;
    findReachingDefs(UseMO, LIS, ReachingDefIndexes);

    // If we find a use that contains this DefMI in its reachingDefs, then it is
    // a reaching use.
    if (any_of(ReachingDefIndexes, [DefIdx](SlotIndex RDIdx) {
          return SlotIndex::isSameInstr(RDIdx, DefIdx);
        }))
      ReachingUses.push_back(&UseMO);
  }
}

bool RewriteMFMAFormStage::initGCNSchedStage() {
  // We only need to run this pass if the architecture supports AGPRs.
  // Additionally, we don't use AGPRs at occupancy levels above 1 so there
  // is no need for this pass in that case, either.
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX90AInsts() || MFI.getMinWavesPerEU() > 1)
    return false;

  RegionsWithExcessArchVGPR.resize(DAG.Regions.size());
  RegionsWithExcessArchVGPR.reset();
  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    GCNRegPressure PressureBefore = DAG.Pressure[Region];
    if (PressureBefore.getArchVGPRNum() > ST.getAddressableNumArchVGPRs())
      RegionsWithExcessArchVGPR[Region] = true;
  }

  if (RegionsWithExcessArchVGPR.none())
    return false;

  TII = ST.getInstrInfo();
  SRI = ST.getRegisterInfo();

  std::vector<std::pair<MachineInstr *, unsigned>> RewriteCands;
  DenseMap<MachineBasicBlock *, std::set<Register>> CopyForUse;
  SmallPtrSet<MachineInstr *, 8> CopyForDef;

  if (!initHeuristics(RewriteCands, CopyForUse, CopyForDef))
    return false;

  int64_t Cost = getRewriteCost(RewriteCands, CopyForUse, CopyForDef);

  // If we haven't found the beneficial conditions, prefer the VGPR form which
  // may result in less cross RC copies.
  if (Cost > 0)
    return false;

  return rewrite(RewriteCands);
}

bool UnclusteredHighRPStage::initGCNSchedStage() {
  if (DisableUnclusterHighRP)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  if (DAG.RegionsWithHighRP.none() && DAG.RegionsWithExcessRP.none())
    return false;

  SavedMutations.swap(DAG.Mutations);
  DAG.addMutation(
      createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PreRAReentry));

  InitialOccupancy = DAG.MinOccupancy;
  // Aggressively try to reduce register pressure in the unclustered high RP
  // stage. Temporarily increase occupancy target in the region.
  TempTargetOccupancy = MFI.getMaxWavesPerEU() > DAG.MinOccupancy
                            ? InitialOccupancy + 1
                            : InitialOccupancy;
  IsAnyRegionScheduled = false;
  S.SGPRLimitBias = S.HighRPSGPRBias;
  S.VGPRLimitBias = S.HighRPVGPRBias;

  LLVM_DEBUG(
      dbgs()
      << "Retrying function scheduling without clustering. "
         "Aggressively try to reduce register pressure to achieve occupancy "
      << TempTargetOccupancy << ".\n");

  return true;
}

bool ClusteredLowOccStage::initGCNSchedStage() {
  if (DisableClusteredLowOccupancy)
    return false;

  if (!GCNSchedStage::initGCNSchedStage())
    return false;

  // Don't bother trying to improve ILP in lower RP regions if occupancy has not
  // been dropped. All regions will have already been scheduled with the ideal
  // occupancy targets.
  if (DAG.StartingOccupancy <= DAG.MinOccupancy)
    return false;

  LLVM_DEBUG(
      dbgs() << "Retrying function scheduling with lowest recorded occupancy "
             << DAG.MinOccupancy << ".\n");
  return true;
}

bool PreRARematStage::isDead(MachineInstr *MI) {
  // Instructions without side-effects are dead iff they only define dead regs.
  // This function is hot and this loop returns early in the common case,
  // so only perform additional checks before this if absolutely necessary.
  for (const MachineOperand &MO : MI->all_defs()) {
    Register Reg = MO.getReg();
    if (Reg.isPhysical()) {
      return false;
    } else {
      if (MO.isDead()) {
        continue;
      }
      for (const MachineInstr &Use : DAG.MRI.use_nodbg_instructions(Reg)) {
        if (&Use != MI) {
          // This def has a non-debug use. Don't delete the instruction!
          return false;
        }
      }
    }
  }

  // Technically speaking inline asm without side effects and no defs can still
  // be deleted. But there is so much bad inline asm code out there, we should
  // let them be.
  if (MI->isInlineAsm())
    return false;

  // FIXME: See issue #105950 for why LIFETIME markers are considered dead here.
  if (MI->isLifetimeMarker())
    return true;

  // If there are no defs with uses, the instruction might be dead.
  return MI->wouldBeTriviallyDead();
}

bool PreRARematStage::eliminateDeadMI() {
  bool AnyChanges = false;

  // Loop over all instructions in all blocks, from bottom to top, so that it's
  // more likely that chains of dependent but ultimately dead instructions will
  // be cleaned up.
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    // Now scan the instructions and delete dead ones, tracking physreg
    // liveness as we go.
    for (MachineInstr &MI : make_early_inc_range(reverse(*MBB))) {
      // If the instruction is dead, delete it!
      if (!MI.hasUnmodeledSideEffects() && !MI.isKill() && isDead(&MI)) {
        LLVM_DEBUG(dbgs() << "DeadMachineInstructionElim: DELETING: " << MI);
        // It is possible that some DBG_VALUE instructions refer to this
        // instruction. They will be deleted in the live debug variable
        // analysis.
        DAG.updateRegionBoundaries(DAG.Regions, MI, nullptr);
        Register Reg = MI.getOperand(0).getReg();
        DAG.LIS->RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
        if (Reg.isVirtual()) {
          DAG.LIS->removeInterval(Reg);
          DAG.LIS->createAndComputeVirtRegInterval(Reg);
        }
        AnyChanges = true;
        continue;
      }
    }
  }

  // Catch all for MBB ranges, subreg liveness, etc..
  DAG.LIS->reanalyze(MF);

  return AnyChanges;
}

bool PreRARematStage::initGCNSchedStage() {
  if (!GCNSchedStage::initGCNSchedStage() || !RematLiveThru)
    return false;

  if (DAG.Regions.size() == 1)
    return false;

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Check maximum occupancy
  if (ST.computeOccupancy(MF.getFunction(), MFI.getLDSSize()).first ==
      DAG.MinOccupancy)
    return false;

  // FIXME: This pass will invalidate cached MBBLiveIns for regions
  // inbetween the defs and region we sinked the def to. Cached pressure
  // for regions where a def is sinked from will also be invalidated. Will
  // need to be fixed if there is another pass after this pass.
  assert(!S.hasNextStage());

  CI.clear();
  CI.compute(MF);
  PDT.recalculate(MF);

  collectRematSeeds();
  if (Cands.empty())
    return false;

  if (!createRematPlan()) {
    ;
  }

  if (!implementRematPlan(TII)) {
    return false;
  }

  bool NeedAggressive = false;
  for (unsigned I = 0; I < OptRegionRPReduction.size(); I++) {
    assert(LiveThruBias >= LiveInBias);
    if (OptRegionRPReduction[I] > (int)(LiveThruBias - LiveInBias)) {
      NeedAggressive = true;
      break;
    }
  }

  if (NeedAggressive && RematLiveIn) {
    DAG.BBLiveInMap = DAG.getRegionLiveInMap();
    DAG.RegionLiveOuts.buildLiveRegMap();

    collectRematSeeds(true);
    bool GoToNext = true;
    if (Cands.empty()) {
      GoToNext = false;
    }

    if (GoToNext && !createRematPlan(true)) {
      GoToNext = false;
    }

    if (GoToNext && !implementRematPlan(TII, true)) {
      GoToNext = false;
    }
  }

  LLVM_DEBUG(
      dbgs() << "Retrying function scheduling with improved occupancy of "
             << DAG.MinOccupancy << " from rematerializing\n");

  eliminateDeadMI();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.isDef())
          continue;
        auto UseReg = MO.getReg();
        if (!UseReg.isVirtual())
          continue;

        DAG.LIS->removeInterval(UseReg);
        DAG.LIS->createAndComputeVirtRegInterval(UseReg);
      }
    }
  }
  return true;
}

void GCNSchedStage::finalizeGCNSchedStage() {
  DAG.finishBlock();
  LLVM_DEBUG(dbgs() << "Ending scheduling stage: " << StageID << "\n");
}

void UnclusteredHighRPStage::finalizeGCNSchedStage() {
  SavedMutations.swap(DAG.Mutations);
  S.SGPRLimitBias = S.VGPRLimitBias = 0;
  if (DAG.MinOccupancy > InitialOccupancy) {
    assert(IsAnyRegionScheduled);
    LLVM_DEBUG(dbgs() << StageID
                      << " stage successfully increased occupancy to "
                      << DAG.MinOccupancy << '\n');
  } else if (!IsAnyRegionScheduled) {
    assert(DAG.MinOccupancy == InitialOccupancy);
    LLVM_DEBUG(dbgs() << StageID
                      << ": No regions scheduled, min occupancy stays at "
                      << DAG.MinOccupancy << ", MFI occupancy stays at "
                      << MFI.getOccupancy() << ".\n");
  }

  GCNSchedStage::finalizeGCNSchedStage();
}

bool GCNSchedStage::initGCNRegion() {
  // Check whether this new region is also a new block.
  if (DAG.RegionBegin->getParent() != CurrentMBB)
    setupNewBlock();

  unsigned NumRegionInstrs = std::distance(DAG.begin(), DAG.end());
  DAG.enterRegion(CurrentMBB, DAG.begin(), DAG.end(), NumRegionInstrs);

  // Skip empty scheduling regions (0 or 1 schedulable instructions).
  if (DAG.begin() == DAG.end() || DAG.begin() == std::prev(DAG.end()))
    return false;

  LLVM_DEBUG(dbgs() << "********** MI Scheduling **********\n");
  LLVM_DEBUG(dbgs() << MF.getName() << ":" << printMBBReference(*CurrentMBB)
                    << " " << CurrentMBB->getName()
                    << "\n  From: " << *DAG.begin() << "    To: ";
             if (DAG.RegionEnd != CurrentMBB->end()) dbgs() << *DAG.RegionEnd;
             else dbgs() << "End";
             dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

  // Save original instruction order before scheduling for possible revert.
  Unsched.clear();
  Unsched.reserve(DAG.NumRegionInstrs);
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule) {
    for (auto &I : DAG) {
      Unsched.push_back(&I);
      if (I.getOpcode() == AMDGPU::SCHED_GROUP_BARRIER ||
          I.getOpcode() == AMDGPU::IGLP_OPT)
        DAG.RegionsWithIGLPInstrs[RegionIdx] = true;
    }
  } else {
    for (auto &I : DAG)
      Unsched.push_back(&I);
  }

  PressureBefore = DAG.Pressure[RegionIdx];
  S.CustomResTracking = DAG.RegionsWithIGLPInstrs[RegionIdx];
  if (S.CustomResTracking)
    S.XDLProcRes.reset();

  LLVM_DEBUG(
      dbgs() << "Pressure before scheduling:\nRegion live-ins:"
             << print(DAG.LiveIns[RegionIdx], DAG.MRI)
             << "Region live-in pressure:  "
             << print(llvm::getRegPressure(DAG.MRI, DAG.LiveIns[RegionIdx]))
             << "Region register pressure: " << print(PressureBefore));

  S.HasHighPressure = false;
  S.KnownExcessRP = isRegionWithExcessRP();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule) {
    SavedMutations.clear();
    SavedMutations.swap(DAG.Mutations);
    bool IsInitialStage = StageID == GCNSchedStageID::OccInitialSchedule ||
                          StageID == GCNSchedStageID::ILPInitialSchedule;
    DAG.addMutation(createIGroupLPDAGMutation(
        IsInitialStage ? AMDGPU::SchedulingPhase::Initial
                       : AMDGPU::SchedulingPhase::PreRAReentry));
  }

  return true;
}

bool UnclusteredHighRPStage::initGCNRegion() {
  // Only reschedule regions that have excess register pressure (i.e. spilling)
  // or had minimum occupancy at the beginning of the stage (as long as
  // rescheduling of previous regions did not make occupancy drop back down to
  // the initial minimum).
  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();
  // If no region has been scheduled yet, the DAG has not yet been updated with
  // the occupancy target. So retrieve it from the temporary.
  unsigned CurrentTargetOccupancy =
      IsAnyRegionScheduled ? DAG.MinOccupancy : TempTargetOccupancy;
  if (!DAG.RegionsWithExcessRP[RegionIdx] &&
      (CurrentTargetOccupancy <= InitialOccupancy ||
       DAG.Pressure[RegionIdx].getOccupancy(ST, DynamicVGPRBlockSize) !=
           InitialOccupancy))
    return false;

  bool IsSchedulingThisRegion = GCNSchedStage::initGCNRegion();
  // If this is the first region scheduled during this stage, make the target
  // occupancy changes in the DAG and MFI.
  if (!IsAnyRegionScheduled && IsSchedulingThisRegion) {
    IsAnyRegionScheduled = true;
    if (MFI.getMaxWavesPerEU() > DAG.MinOccupancy) {
      DAG.MinOccupancy = TempTargetOccupancy;
      MFI.increaseOccupancy(MF, TempTargetOccupancy);
    }
  }
  return IsSchedulingThisRegion;
}

bool ClusteredLowOccStage::initGCNRegion() {
  // We may need to reschedule this region if it wasn't rescheduled in the last
  // stage, or if we found it was testing critical register pressure limits in
  // the unclustered reschedule stage. The later is because we may not have been
  // able to raise the min occupancy in the previous stage so the region may be
  // overly constrained even if it was already rescheduled.
  if (!DAG.RegionsWithHighRP[RegionIdx])
    return false;

  return GCNSchedStage::initGCNRegion();
}

bool PreRARematStage::initGCNRegion() {
  return RescheduleRegions[RegionIdx] && GCNSchedStage::initGCNRegion();
}

void GCNSchedStage::setupNewBlock() {
  if (CurrentMBB)
    DAG.finishBlock();

  CurrentMBB = DAG.RegionBegin->getParent();
  DAG.startBlock(CurrentMBB);
  // Get real RP for the region if it hasn't be calculated before. After the
  // initial schedule stage real RP will be collected after scheduling.
  if (StageID == GCNSchedStageID::OccInitialSchedule ||
      StageID == GCNSchedStageID::ILPInitialSchedule)
    DAG.computeBlockPressure(RegionIdx, CurrentMBB);
}

void GCNSchedStage::finalizeGCNRegion() {
  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
  if (S.HasHighPressure)
    DAG.RegionsWithHighRP[RegionIdx] = true;

  // Revert scheduling if we have dropped occupancy or there is some other
  // reason that the original schedule is better.
  checkScheduling();

  if (DAG.RegionsWithIGLPInstrs[RegionIdx] &&
      StageID != GCNSchedStageID::UnclusteredHighRPReschedule)
    SavedMutations.swap(DAG.Mutations);

  DAG.exitRegion();
  advanceRegion();
}

void GCNSchedStage::checkScheduling() {
  // Check the results of scheduling.
  PressureAfter = DAG.getRealRegPressure(RegionIdx);

  LLVM_DEBUG(dbgs() << "Pressure after scheduling: " << print(PressureAfter));
  LLVM_DEBUG(dbgs() << "Region: " << RegionIdx << ".\n");

  unsigned DynamicVGPRBlockSize = DAG.MFI.getDynamicVGPRBlockSize();

  if (PressureAfter.getSGPRNum() <= S.SGPRCriticalLimit &&
      PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) <= S.VGPRCriticalLimit) {
    DAG.Pressure[RegionIdx] = PressureAfter;

    // Early out if we have achieved the occupancy target.
    LLVM_DEBUG(dbgs() << "Pressure in desired limits, done.\n");
    return;
  }

  unsigned TargetOccupancy = std::min(
      S.getTargetOccupancy(), ST.getOccupancyWithWorkGroupSizes(MF).second);
  unsigned WavesAfter = std::min(
      TargetOccupancy, PressureAfter.getOccupancy(ST, DynamicVGPRBlockSize));
  unsigned WavesBefore = std::min(
      TargetOccupancy, PressureBefore.getOccupancy(ST, DynamicVGPRBlockSize));
  LLVM_DEBUG(dbgs() << "Occupancy before scheduling: " << WavesBefore
                    << ", after " << WavesAfter << ".\n");

  // We may not be able to keep the current target occupancy because of the just
  // scheduled region. We might still be able to revert scheduling if the
  // occupancy before was higher, or if the current schedule has register
  // pressure higher than the excess limits which could lead to more spilling.
  unsigned NewOccupancy = std::max(WavesAfter, WavesBefore);

  // Allow memory bound functions to drop to 4 waves if not limited by an
  // attribute.
  if (WavesAfter < WavesBefore && WavesAfter < DAG.MinOccupancy &&
      WavesAfter >= MFI.getMinAllowedOccupancy()) {
    LLVM_DEBUG(dbgs() << "Function is memory bound, allow occupancy drop up to "
                      << MFI.getMinAllowedOccupancy() << " waves\n");
    NewOccupancy = WavesAfter;
  }

  if (NewOccupancy < DAG.MinOccupancy) {
    DAG.MinOccupancy = NewOccupancy;
    MFI.limitOccupancy(DAG.MinOccupancy);
    LLVM_DEBUG(dbgs() << "Occupancy lowered for the function to "
                      << DAG.MinOccupancy << ".\n");
  }
  // The maximum number of arch VGPR on non-unified register file, or the
  // maximum VGPR + AGPR in the unified register file case.
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  // The maximum number of arch VGPR for both unified and non-unified register
  // file.
  unsigned MaxArchVGPRs = std::min(MaxVGPRs, ST.getAddressableNumArchVGPRs());
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);

  if (PressureAfter.getVGPRNum(ST.hasGFX90AInsts()) > MaxVGPRs ||
      PressureAfter.getArchVGPRNum() > MaxArchVGPRs ||
      PressureAfter.getAGPRNum() > MaxArchVGPRs ||
      PressureAfter.getSGPRNum() > MaxSGPRs) {
    DAG.RegionsWithHighRP[RegionIdx] = true;
    DAG.RegionsWithExcessRP[RegionIdx] = true;
  }

  // Revert if this region's schedule would cause a drop in occupancy or
  // spilling.
  if (shouldRevertScheduling(WavesAfter))
    revertScheduling();
  else
    DAG.Pressure[RegionIdx] = PressureAfter;
}

unsigned
GCNSchedStage::computeSUnitReadyCycle(const SUnit &SU, unsigned CurrCycle,
                                      DenseMap<unsigned, unsigned> &ReadyCycles,
                                      const TargetSchedModel &SM) {
  unsigned ReadyCycle = CurrCycle;
  for (auto &D : SU.Preds) {
    if (D.isAssignedRegDep()) {
      MachineInstr *DefMI = D.getSUnit()->getInstr();
      unsigned Latency = SM.computeInstrLatency(DefMI);
      unsigned DefReady = ReadyCycles[DAG.getSUnit(DefMI)->NodeNum];
      ReadyCycle = std::max(ReadyCycle, DefReady + Latency);
    }
  }
  ReadyCycles[SU.NodeNum] = ReadyCycle;
  return ReadyCycle;
}

#ifndef NDEBUG
struct EarlierIssuingCycle {
  bool operator()(std::pair<MachineInstr *, unsigned> A,
                  std::pair<MachineInstr *, unsigned> B) const {
    return A.second < B.second;
  }
};

static void printScheduleModel(std::set<std::pair<MachineInstr *, unsigned>,
                                        EarlierIssuingCycle> &ReadyCycles) {
  if (ReadyCycles.empty())
    return;
  unsigned BBNum = ReadyCycles.begin()->first->getParent()->getNumber();
  dbgs() << "\n################## Schedule time ReadyCycles for MBB : " << BBNum
         << " ##################\n# Cycle #\t\t\tInstruction          "
            "             "
            "                            \n";
  unsigned IPrev = 1;
  for (auto &I : ReadyCycles) {
    if (I.second > IPrev + 1)
      dbgs() << "****************************** BUBBLE OF " << I.second - IPrev
             << " CYCLES DETECTED ******************************\n\n";
    dbgs() << "[ " << I.second << " ]  :  " << *I.first << "\n";
    IPrev = I.second;
  }
}
#endif

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const std::vector<SUnit> &InputSchedule) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &SU : InputSchedule) {
    unsigned ReadyCycle =
        computeSUnitReadyCycle(SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU.getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

ScheduleMetrics
GCNSchedStage::getScheduleMetrics(const GCNScheduleDAGMILive &DAG) {
#ifndef NDEBUG
  std::set<std::pair<MachineInstr *, unsigned>, EarlierIssuingCycle>
      ReadyCyclesSorted;
#endif
  const TargetSchedModel &SM = ST.getInstrInfo()->getSchedModel();
  unsigned SumBubbles = 0;
  DenseMap<unsigned, unsigned> ReadyCycles;
  unsigned CurrCycle = 0;
  for (auto &MI : DAG) {
    SUnit *SU = DAG.getSUnit(&MI);
    if (!SU)
      continue;
    unsigned ReadyCycle =
        computeSUnitReadyCycle(*SU, CurrCycle, ReadyCycles, SM);
    SumBubbles += ReadyCycle - CurrCycle;
#ifndef NDEBUG
    ReadyCyclesSorted.insert(std::make_pair(SU->getInstr(), ReadyCycle));
#endif
    CurrCycle = ++ReadyCycle;
  }
#ifndef NDEBUG
  LLVM_DEBUG(
      printScheduleModel(ReadyCyclesSorted);
      dbgs() << "\n\t"
             << "Metric: "
             << (SumBubbles
                     ? (SumBubbles * ScheduleMetrics::ScaleFactor) / CurrCycle
                     : 1)
             << "\n\n");
#endif

  return ScheduleMetrics(CurrCycle, SumBubbles);
}

bool GCNSchedStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (S.CustomResTracking)
    return false;

  if (WavesAfter < DAG.MinOccupancy)
    return true;

  // For dynamic VGPR mode, we don't want to waste any VGPR blocks.
  if (DAG.MFI.isDynamicVGPREnabled()) {
    unsigned BlocksBefore = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureBefore.getVGPRNum(false));
    unsigned BlocksAfter = AMDGPU::IsaInfo::getAllocatedNumVGPRBlocks(
        &ST, DAG.MFI.getDynamicVGPRBlockSize(),
        PressureAfter.getVGPRNum(false));
    if (BlocksAfter > BlocksBefore)
      return true;
  }

  return false;
}

bool OccInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool UnclusteredHighRPStage::shouldRevertScheduling(unsigned WavesAfter) {
  // If RP is not reduced in the unclustered reschedule stage, revert to the
  // old schedule.
  if ((WavesAfter <=
           PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()) &&
       mayCauseSpilling(WavesAfter)) ||
      GCNSchedStage::shouldRevertScheduling(WavesAfter)) {
    LLVM_DEBUG(dbgs() << "Unclustered reschedule did not help.\n");
    return true;
  }

  // Do not attempt to relax schedule even more if we are already spilling.
  if (isRegionWithExcessRP())
    return false;

  LLVM_DEBUG(
      dbgs()
      << "\n\t      *** In shouldRevertScheduling ***\n"
      << "      *********** BEFORE UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MBefore =
      getScheduleMetrics(DAG.SUnits);
  LLVM_DEBUG(
      dbgs()
      << "\n      *********** AFTER UnclusteredHighRPStage ***********\n");
  ScheduleMetrics MAfter = getScheduleMetrics(DAG);
  unsigned OldMetric = MBefore.getMetric();
  unsigned NewMetric = MAfter.getMetric();
  unsigned WavesBefore = std::min(
      S.getTargetOccupancy(),
      PressureBefore.getOccupancy(ST, DAG.MFI.getDynamicVGPRBlockSize()));
  unsigned Profit =
      ((WavesAfter * ScheduleMetrics::ScaleFactor) / WavesBefore *
       ((OldMetric + ScheduleMetricBias) * ScheduleMetrics::ScaleFactor) /
       NewMetric) /
      ScheduleMetrics::ScaleFactor;
  LLVM_DEBUG(dbgs() << "\tMetric before " << MBefore << "\tMetric after "
                    << MAfter << "Profit: " << Profit << "\n");
  return Profit < ScheduleMetrics::ScaleFactor;
}

bool ClusteredLowOccStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (PressureAfter == PressureBefore)
    return false;

  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool PreRARematStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (GCNSchedStage::shouldRevertScheduling(WavesAfter))
    return true;

  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool ILPInitialScheduleStage::shouldRevertScheduling(unsigned WavesAfter) {
  if (mayCauseSpilling(WavesAfter))
    return true;

  return false;
}

bool MemoryClauseInitialScheduleStage::shouldRevertScheduling(
    unsigned WavesAfter) {
  return mayCauseSpilling(WavesAfter);
}

bool GCNSchedStage::mayCauseSpilling(unsigned WavesAfter) {
  if (WavesAfter <= MFI.getMinWavesPerEU() && isRegionWithExcessRP() &&
      !PressureAfter.less(MF, PressureBefore)) {
    LLVM_DEBUG(dbgs() << "New pressure will result in more spilling.\n");
    return true;
  }

  return false;
}

void GCNSchedStage::revertScheduling() {
  LLVM_DEBUG(dbgs() << "Attempting to revert scheduling.\n");
  DAG.RegionEnd = DAG.RegionBegin;
  int SkippedDebugInstr = 0;
  for (MachineInstr *MI : Unsched) {
    if (MI->isDebugInstr()) {
      ++SkippedDebugInstr;
      continue;
    }

    if (MI->getIterator() != DAG.RegionEnd) {
      DAG.BB->remove(MI);
      DAG.BB->insert(DAG.RegionEnd, MI);
      if (!MI->isDebugInstr())
        DAG.LIS->handleMove(*MI, true);
    }

    // Reset read-undef flags and update them later.
    for (auto &Op : MI->all_defs())
      Op.setIsUndef(false);
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG.TRI, DAG.MRI, DAG.ShouldTrackLaneMasks, false);
    if (!MI->isDebugInstr()) {
      if (DAG.ShouldTrackLaneMasks) {
        // Adjust liveness and add missing dead+read-undef flags.
        SlotIndex SlotIdx = DAG.LIS->getInstructionIndex(*MI).getRegSlot();
        RegOpers.adjustLaneLiveness(*DAG.LIS, DAG.MRI, SlotIdx, MI);
      } else {
        // Adjust for missing dead-def flags.
        RegOpers.detectDeadDefs(*MI, *DAG.LIS);
      }
    }
    DAG.RegionEnd = MI->getIterator();
    ++DAG.RegionEnd;
    LLVM_DEBUG(dbgs() << "Scheduling " << *MI);
  }

  // After reverting schedule, debug instrs will now be at the end of the block
  // and RegionEnd will point to the first debug instr. Increment RegionEnd
  // pass debug instrs to the actual end of the scheduling region.
  while (SkippedDebugInstr-- > 0)
    ++DAG.RegionEnd;

  // If Unsched.front() instruction is a debug instruction, this will actually
  // shrink the region since we moved all debug instructions to the end of the
  // block. Find the first instruction that is not a debug instruction.
  DAG.RegionBegin = Unsched.front()->getIterator();
  if (DAG.RegionBegin->isDebugInstr()) {
    for (MachineInstr *MI : Unsched) {
      if (MI->isDebugInstr())
        continue;
      DAG.RegionBegin = MI->getIterator();
      break;
    }
  }

  // Then move the debug instructions back into their correct place and set
  // RegionBegin and RegionEnd if needed.
  DAG.placeDebugValues();

  DAG.Regions[RegionIdx] = std::pair(DAG.RegionBegin, DAG.RegionEnd);
}

bool RewriteMFMAFormStage::isRewriteCandidate(MachineInstr *MI) const {

  if (!static_cast<const SIInstrInfo *>(DAG.TII)->isMAI(*MI))
    return false;
  return AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode()) != -1;
}

bool RewriteMFMAFormStage::initHeuristics(
    std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  bool Changed = false;

  // Prepare for the heuristics
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isRewriteCandidate(&MI))
        continue;

      int ReplacementOp = AMDGPU::getMFMASrcCVDstAGPROp(MI.getOpcode());
      assert(ReplacementOp != -1);

      RewriteCands.push_back({&MI, MI.getOpcode()});
      MI.setDesc(TII->get(ReplacementOp));

      MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
      if (Src2->isReg()) {
        SmallVector<SlotIndex, 8> Src2ReachingDefs;
        findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);

        // For any definition of the src2 register which is non-MFMA, we
        // insert a copy.
        for (SlotIndex RDIdx : Src2ReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
          if (!TII->isMAI(*RD))
            CopyForDef.insert(RD);
        }
      }

      MachineOperand &Dst = MI.getOperand(0);
      SmallVector<MachineOperand *, 8> DstReachingUses;

      findReachingUses(&MI, DAG.LIS, DstReachingUses);

      for (MachineOperand *RUOp : DstReachingUses) {
        if (TII->isMAI(*RUOp->getParent()))
          continue;

        // For any user of the result of the MFMA which is not an MFMA, we
        // insert a copy. For a given register, we will only insert one copy
        // per user block.
        CopyForUse[RUOp->getParent()->getParent()].insert(RUOp->getReg());

        SmallVector<SlotIndex, 8> DstUsesReachingDefs;
        findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

        for (SlotIndex RDIndex : DstUsesReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
          if (TII->isMAI(*RD))
            continue;

          // For any definition of the user of the MFMA which is not an MFMA,
          // we insert a copy. We do this to transform all the reaching defs
          // of this use to AGPR. By doing this, we can insert a copy from
          // AGPR to VGPR at the user rather than after the MFMA.
          CopyForDef.insert(RD);
        }
      }

      // Do the rewrite to allow for updated RP calculation.
      const TargetRegisterClass *VGPRRC = DAG.MRI.getRegClass(Dst.getReg());
      const TargetRegisterClass *AGPRRC = SRI->getEquivalentAGPRClass(VGPRRC);
      DAG.MRI.setRegClass(Dst.getReg(), AGPRRC);
      if (Src2->isReg())
        DAG.MRI.setRegClass(Src2->getReg(), AGPRRC);
      Changed = true;
    }
  }

  return Changed;
}

int64_t RewriteMFMAFormStage::getRewriteCost(
    const std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    const DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    const SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  // MBFI and GCNRegPressure::getVGPRSpills are not available at this LLVM
  // version. The stage is disabled by default (DisableRewriteMFMAFormSchedStage
  // = true), so this function is unreachable.
  return 0;
#if 0
  MachineBlockFrequencyInfo *MBFI = DAG.MBFI;

  int64_t BestSpillCost = 0;
  int64_t Cost = 0;
  uint64_t EntryFreq = MBFI->getEntryFreq().getFrequency();

  std::pair<unsigned, unsigned> MaxVectorRegs =
      ST.getMaxNumVectorRegs(MF.getFunction());
  unsigned ArchVGPRThreshold = MaxVectorRegs.first;
  unsigned AGPRThreshold = MaxVectorRegs.second;
  unsigned CombinedThreshold = ST.getMaxNumVGPRs(MF);

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    if (!RegionsWithExcessArchVGPR[Region])
      continue;

    GCNRegPressure &PressureBefore = DAG.Pressure[Region];
    unsigned SpillCostBefore = PressureBefore.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);

    // For the cases we care about (i.e. ArchVGPR usage is greater than the
    // addressable limit), rewriting alone should bring pressure to manageable
    // level. If we find any such region, then the rewrite is potentially
    // beneficial.
    GCNRegPressure PressureAfter = DAG.getRealRegPressure(Region);
    unsigned SpillCostAfter = PressureAfter.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);

    uint64_t BlockFreq =
        MBFI->getBlockFreq(DAG.Regions[Region].first->getParent())
            .getFrequency();

    bool RelativeFreqIsDenom = EntryFreq > BlockFreq;
    uint64_t RelativeFreq = EntryFreq && BlockFreq
                                ? (RelativeFreqIsDenom ? EntryFreq / BlockFreq
                                                       : BlockFreq / EntryFreq)
                                : 1;

    // This assumes perfect spilling / splitting -- using one spill / copy
    // instruction and one restoreFrom / copy for each excess register,
    int64_t SpillCost = ((int)SpillCostAfter - (int)SpillCostBefore) * 2;

    // Also account for the block frequency.
    if (RelativeFreqIsDenom)
      SpillCost /= (int64_t)RelativeFreq;
    else
      SpillCost *= (int64_t)RelativeFreq;

    // If we have increased spilling in any block, just bail.
    if (SpillCost > 0)
      return SpillCost;

    if (SpillCost < BestSpillCost)
      BestSpillCost = SpillCost;
  }

  // Set the cost to the largest decrease in spill cost in order to not double
  // count spill reductions.
  Cost = BestSpillCost;
  assert(Cost <= 0);

  unsigned CopyCost = 0;

  // For each CopyForDef, increase the cost by the register size while
  // accounting for block frequency.
  for (MachineInstr *DefMI : CopyForDef) {
    Register DefReg = DefMI->getOperand(0).getReg();
    uint64_t DefFreq =
        EntryFreq
            ? MBFI->getBlockFreq(DefMI->getParent()).getFrequency() / EntryFreq
            : 1;

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(DefReg);
    CopyCost += RC->getCopyCost() * DefFreq;
  }

  // Account for CopyForUse copies in each block that the register is used.
  for (auto &[UseBlock, UseRegs] : CopyForUse) {
    uint64_t UseFreq =
        EntryFreq ? MBFI->getBlockFreq(UseBlock).getFrequency() / EntryFreq : 1;

    for (Register UseReg : UseRegs) {
      const TargetRegisterClass *RC = DAG.MRI.getRegClass(UseReg);
      CopyCost += RC->getCopyCost() * UseFreq;
    }
  }

  // Reset the classes that were changed to AGPR for better RB analysis.
  // We must do rewriting after copy-insertion, as some defs of the register
  // may require VGPR.  Additionally, if we bail out and don't perform the
  // rewrite then these need to be restored anyway.
  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    assert(TII->isMAI(*MI));
    const TargetRegisterClass *AGPRRC =
        DAG.MRI.getRegClass(MI->getOperand(0).getReg());
    const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(AGPRRC);

    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    assert(Src2);

    if (Src2->isReg())
      DAG.MRI.setRegClass(Src2->getReg(), VGPRRC);
    DAG.MRI.setRegClass(MI->getOperand(0).getReg(), VGPRRC);
    MI->setDesc(TII->get(OriginalOpcode));
  }

  return Cost + CopyCost;
#endif
}

bool RewriteMFMAFormStage::rewrite(
    const std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands) {
  DenseMap<MachineInstr *, unsigned> FirstMIToRegion;
  DenseMap<MachineInstr *, unsigned> LastMIToRegion;

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    RegionBoundaries Entry = DAG.Regions[Region];
    if (Entry.first == Entry.second)
      continue;

    FirstMIToRegion[&*Entry.first] = Region;
    if (Entry.second != Entry.first->getParent()->end())
      LastMIToRegion[&*Entry.second] = Region;
  }

  // Rewrite the MFMAs to AGPR, and insert any copies as needed.
  // The general assumption of the algorithm (and the previous cost calculation)
  // is that it is better to insert the copies in the MBB of the def of the src2
  // operands, and in the MBB of the user of the dest operands. This is based on
  // the assumption that the MFMAs are likely to appear in loop bodies, while
  // the src2 and dest operands are live-in / live-out of the loop. Due to this
  // design, the algorithm for finding copy insertion points is more
  // complicated.
  //
  // There are three main cases to handle: 1. the reaching defs of the src2
  // operands, 2. the reaching uses of the dst operands, and 3. the reaching
  // defs of the reaching uses of the dst operand.
  //
  // In the first case, we simply insert copies after each of the reaching
  // definitions. In the second case, we collect all the uses of a given dest
  // and organize them by MBB. Then, we insert 1 copy for each MBB before the
  // earliest use. Since the use may have multiple reaching defs, and since we
  // want to replace the register it is using with the result of the copy, we
  // must handle case 3. In the third case, we simply insert a copy after each
  // of the reaching defs to connect to the copy of the reaching uses of the dst
  // reg. This allows us to avoid inserting copies next to the MFMAs.
  //
  // While inserting the copies, we maintain a map of operands which will use
  // different regs (i.e. the result of the copies). For example, a case 1 src2
  // operand will use the register result of the copies after the reaching defs,
  // as opposed to the original register. Now that we have completed our copy
  // analysis and placement, we can bulk update the registers. We do this
  // separately as to avoid complicating the reachingDef and reachingUse
  // queries.
  //
  // While inserting the copies, we also maintain a list or registers which we
  // will want to reclassify as AGPR. After doing the copy insertion and the
  // register replacement, we can finally do the reclassification. This uses the
  // redef map, as the registers we are interested in reclassifying may be
  // replaced by the result of a copy. We must do this after the copy analysis
  // and placement as we must have an accurate redef map -- otherwise we may end
  // up creating illegal instructions.

  // The original registers of the MFMA that need to be reclassified as AGPR.
  DenseSet<Register> RewriteRegs;
  // The map of an original register in the MFMA to a new register (result of a
  // copy) that it should be replaced with.
  DenseMap<Register, Register> RedefMap;
  // The map of the original MFMA registers to the relevant MFMA operands.
  DenseMap<Register, DenseSet<MachineOperand *>> ReplaceMap;
  // The map of reaching defs for a given register -- to avoid duplicate copies.
  DenseMap<Register, SmallPtrSet<MachineInstr *, 8>> ReachingDefCopyMap;
  // The map of reaching uses for a given register by basic block -- to avoid
  // duplicate copies and to calculate per MBB insert pts.
  DenseMap<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>
      ReachingUseTracker;

  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    int ReplacementOp = AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode());
    if (ReplacementOp == -1)
      continue;
    MI->setDesc(TII->get(ReplacementOp));

    // Case 1: insert copies for the reaching defs of the Src2Reg.
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2->isReg()) {
      Register Src2Reg = Src2->getReg();
      if (!Src2Reg.isVirtual())
        return false;

      Register MappedReg = Src2->getReg();
      SmallVector<SlotIndex, 8> Src2ReachingDefs;
      findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);
      SmallSetVector<MachineInstr *, 8> Src2DefsReplace;

      for (SlotIndex RDIndex : Src2ReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (TII->isMAI(*RD))
          continue;

        // If there is a non mai reaching def, then we need a copy.
        Src2DefsReplace.insert(RD);
      }

      if (!Src2DefsReplace.empty()) {
        DenseMap<Register, Register>::iterator RI = RedefMap.find(Src2Reg);
        if (RI != RedefMap.end()) {
          MappedReg = RI->second;
        } else {
          assert(!ReachingDefCopyMap.contains(Src2Reg));
          const TargetRegisterClass *Src2RC = DAG.MRI.getRegClass(Src2Reg);
          const TargetRegisterClass *VGPRRC =
              SRI->getEquivalentVGPRClass(Src2RC);

          // Track the mapping of the original register to the new register.
          MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
          RedefMap[Src2Reg] = MappedReg;
        }

        // If none exists, create a copy from this reaching def.
        // We may have inserted a copy already in an earlier iteration.
        for (MachineInstr *RD : Src2DefsReplace) {
          // Do not create redundant copies.
          if (ReachingDefCopyMap[Src2Reg].insert(RD).second) {
            MachineInstrBuilder VGPRCopy =
                BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                        RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                    .addDef(MappedReg, {}, 0)
                    .addUse(Src2Reg, {}, 0);
            DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

            // If this reaching def was the last MI in the region, update the
            // region boundaries.
            if (LastMIToRegion.contains(RD)) {
              unsigned UpdateRegion = LastMIToRegion[RD];
              DAG.Regions[UpdateRegion].second = VGPRCopy;
              LastMIToRegion.erase(RD);
            }
          }
        }
      }

      // Track the register for reclassification
      RewriteRegs.insert(Src2Reg);

      // Always insert the operand for replacement. If this corresponds with a
      // chain of tied-def we may not see the VGPR requirement until later.
      ReplaceMap[Src2Reg].insert(Src2);
    }

    // Case 2 and Case 3: insert copies before the reaching uses of the dsts,
    // and after the reaching defs of the reaching uses of the dsts.

    MachineOperand *Dst = &MI->getOperand(0);
    Register DstReg = Dst->getReg();
    if (!DstReg.isVirtual())
      return false;

    Register MappedReg = DstReg;
    SmallVector<MachineOperand *, 8> DstReachingUses;

    SmallVector<MachineOperand *, 8> DstReachingUseCopies;
    SmallVector<MachineInstr *, 8> DstUseDefsReplace;

    findReachingUses(MI, DAG.LIS, DstReachingUses);

    for (MachineOperand *RUOp : DstReachingUses) {
      if (TII->isMAI(*RUOp->getParent()))
        continue;

      // If there is a non mai reaching use, then we need a copy.
      if (find(DstReachingUseCopies, RUOp) == DstReachingUseCopies.end())
        DstReachingUseCopies.push_back(RUOp);
      SmallVector<SlotIndex, 8> DstUsesReachingDefs;
      findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

      for (SlotIndex RDIndex : DstUsesReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (TII->isMAI(*RD))
          continue;

        // If there is a non mai reaching def of this reaching use, then we will
        // need a copy.
        if (find(DstUseDefsReplace, RD) == DstUseDefsReplace.end())
          DstUseDefsReplace.push_back(RD);
      }
    }

    if (!DstUseDefsReplace.empty()) {
      DenseMap<Register, Register>::iterator RI = RedefMap.find(DstReg);
      if (RI != RedefMap.end()) {
        MappedReg = RI->second;
      } else {
        assert(!ReachingDefCopyMap.contains(DstReg));
        const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
        const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);

        // Track the mapping of the original register to the new register.
        MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
        RedefMap[DstReg] = MappedReg;
      }

      // If none exists, create a copy from this reaching def.
      // We may have inserted a copy already in an earlier iteration.
      for (MachineInstr *RD : DstUseDefsReplace) {
        // Do not create reundant copies.
        if (ReachingDefCopyMap[DstReg].insert(RD).second) {
          MachineInstrBuilder VGPRCopy =
              BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                      RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                  .addDef(MappedReg, {}, 0)
                  .addUse(DstReg, {}, 0);
          DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

          // If this reaching def was the last MI in the region, update the
          // region boundaries.
          DenseMap<MachineInstr *, unsigned>::iterator LMI =
              LastMIToRegion.find(RD);
          if (LMI != LastMIToRegion.end()) {
            unsigned UpdateRegion = LMI->second;
            DAG.Regions[UpdateRegion].second = VGPRCopy;
            LastMIToRegion.erase(RD);
          }
        }
      }
    }

    DenseSet<MachineOperand *> &DstRegSet = ReplaceMap[DstReg];
    for (MachineOperand *RU : DstReachingUseCopies) {
      MachineBasicBlock *RUBlock = RU->getParent()->getParent();
      // Just keep track of the reaching use of this register by block. After we
      // have scanned all the MFMAs we can find optimal insert pts.
      if (RUBlock != MI->getParent()) {
        ReachingUseTracker[RUBlock->getNumber()][DstReg].insert(RU);
        continue;
      }

      // Special case, the use is in the same block as the MFMA. Insert the copy
      // just before the use.
      const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
      const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
      Register NewUseReg = DAG.MRI.createVirtualRegister(VGPRRC);
      MachineInstr *UseInst = RU->getParent();
      MachineInstrBuilder VGPRCopy =
          BuildMI(*UseInst->getParent(), UseInst->getIterator(),
                  UseInst->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(NewUseReg, {}, 0)
              .addUse(DstReg, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);
      // Since we know this use has only one reaching def, we can replace the
      // use reg.
      RU->setReg(NewUseReg);
      // Track the copy source operand for r eplacement.
      DstRegSet.insert(&VGPRCopy->getOperand(1));
    }

    // Track the register for reclassification
    RewriteRegs.insert(DstReg);

    // Insert the dst operand for replacement. If this dst is in a chain of
    // tied-def MFMAs, and the first src2 needs to be replaced with a new reg,
    // all the correspond operands need to be replaced.
    DstRegSet.insert(Dst);
  }

  // Handle the copies for dst uses.
  using RUBType =
      std::pair<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>;
  for (RUBType RUBlockEntry : ReachingUseTracker) {
    using RUDType = std::pair<Register, SmallPtrSet<MachineOperand *, 8>>;
    for (RUDType RUDst : RUBlockEntry.second) {
      MachineOperand *OpBegin = *RUDst.second.begin();
      SlotIndex InstPt = DAG.LIS->getInstructionIndex(*OpBegin->getParent());

      // Find the earliest use in this block.
      for (MachineOperand *User : RUDst.second) {
        SlotIndex NewInstPt = DAG.LIS->getInstructionIndex(*User->getParent());
        if (SlotIndex::isEarlierInstr(NewInstPt, InstPt))
          InstPt = NewInstPt;
      }

      const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(RUDst.first);
      const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
      Register NewUseReg = DAG.MRI.createVirtualRegister(VGPRRC);
      MachineInstr *UseInst = DAG.LIS->getInstructionFromIndex(InstPt);

      MachineInstrBuilder VGPRCopy =
          BuildMI(*UseInst->getParent(), UseInst->getIterator(),
                  UseInst->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(NewUseReg, {}, 0)
              .addUse(RUDst.first, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

      // If this UseInst was the first MI in the region, update the region
      // boundaries.
      DenseMap<MachineInstr *, unsigned>::iterator FI =
          FirstMIToRegion.find(UseInst);
      if (FI != FirstMIToRegion.end()) {
        unsigned UpdateRegion = FI->second;
        DAG.Regions[UpdateRegion].first = VGPRCopy;
        FirstMIToRegion.erase(UseInst);
      }

      // Replace the operand for all users.
      for (MachineOperand *User : RUDst.second) {
        User->setReg(NewUseReg);
      }

      // Track the copy source operand for replacement.
      ReplaceMap[RUDst.first].insert(&VGPRCopy->getOperand(1));
    }
  }

  // We may have needed to insert copies after the reaching defs of the MFMAs.
  // Replace the original register with the result of the copy for all relevant
  // operands.
  for (std::pair<Register, Register> NewDef : RedefMap) {
    Register OldReg = NewDef.first;
    Register NewReg = NewDef.second;

    // Replace the register for any associated operand in the MFMA chain.
    for (MachineOperand *ReplaceOp : ReplaceMap[OldReg])
      ReplaceOp->setReg(NewReg);
  }

  // Finally, do the reclassification of the MFMA registers.
  for (Register RewriteReg : RewriteRegs) {
    Register RegToRewrite = RewriteReg;

    // Be sure to update the replacement register and not the original.
    DenseMap<Register, Register>::iterator RI = RedefMap.find(RewriteReg);
    if (RI != RedefMap.end())
      RegToRewrite = RI->second;

    const TargetRegisterClass *CurrRC = DAG.MRI.getRegClass(RegToRewrite);
    const TargetRegisterClass *AGPRRC = SRI->getEquivalentAGPRClass(CurrRC);

    DAG.MRI.setRegClass(RegToRewrite, AGPRRC);
  }

  // Bulk update the LIS.
  DAG.LIS->reanalyze(DAG.MF);
  // Liveins may have been modified for cross RC copies
  RegionPressureMap LiveInUpdater(&DAG, false);
  LiveInUpdater.buildLiveRegMap();

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++)
    DAG.LiveIns[Region] = LiveInUpdater.getLiveRegsForRegionIdx(Region);

  DAG.Pressure[RegionIdx] = DAG.getRealRegPressure(RegionIdx);

  return true;
}

bool PreRARematStage::canRemat(Register Reg) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(DAG.TRI);
  if (!DAG.LIS->hasInterval(Reg))
    return false;

  // TODO: Handle AGPR and SGPR rematerialization
  if (!SRI->isVGPRClass(DAG.MRI.getRegClass(Reg)) || !DAG.MRI.hasOneDef(Reg)) {
    return false;
  }

  MachineOperand *Op = DAG.MRI.getOneDef(Reg);
  MachineInstr *Def = Op->getParent();

  if (/*Op->getSubReg() != 0 ||*/ !isTriviallyReMaterializable(*Def)) {
    return false;
  }
  return true;
}

Printable print(const RematCandidate &R) {
  return Printable([&R](raw_ostream &OS) {
    OS << "Remat Candidate for Def: ";
    R.Def->dump();
    OS << "Into regions: \n";
    bool NeedsComma = false;
    for (auto Region : R.HighRPRegions) {
      if (NeedsComma)
        OS << ", ";
      OS << Region;
    }
    OS << "\n";
  });
}

void PreRARematStage::collectRematSeeds(bool Aggressive) {
  if (!Aggressive) {
    for (unsigned I = 0, E = DAG.Regions.size(); I != E; I++) {
      auto TheBlock = DAG.Regions[I].first->getParent();
      auto Cycle = CI.getCycle(TheBlock);
      if (Cycle) {
        TheBlock = const_cast<MachineBasicBlock *>(*Cycle->block_begin());
        if (!TargetBlock)
          TargetBlock = TheBlock;
      }
    }
  }

  RelevantRegions.resize(DAG.Regions.size());
  RelevantRegions.reset();
  SmallPtrSet<MachineBasicBlock *, 4> Visited;
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; I++) {
    if (DAG.Regions[I].first->getParent() != TargetBlock)
      continue;

    RelevantRegions[I] = true;

    auto TheBlock = DAG.Regions[I].first->getParent();

    auto TheLiveIns = DAG.LiveIns[I];

    GCNRegPressure LiveThru;
    for (auto LR : TheLiveIns) {
      auto TheReg = LR.first;
      bool AddedToRematList = false;
      bool FoundBlockUse = false;
      if (!Aggressive) {
        for (auto &TheUseInst : DAG.MRI.use_nodbg_instructions(TheReg)) {
          if (TheUseInst.getParent() == TheBlock) {
            FoundBlockUse = true;
            break;
          }
        }
      }
      if (!FoundBlockUse && canRemat(TheReg)) {
        LiveThru.inc(LR.first, (LaneBitmask)0, LR.second, DAG.MRI);
        MachineInstr *Def = DAG.MRI.getOneDef(TheReg)->getParent();
        for (auto &TheUseInst : DAG.MRI.use_nodbg_instructions(TheReg)) {
          if (!Aggressive &&
              !isReachableFrom(TargetBlock, TheUseInst.getParent()))
            continue;
          MachineBasicBlock::iterator InstPt = &TheUseInst;
          RematCandidate R(Def, CI.getCycleDepth(InstPt->getParent()), I,
                           InstPt);
          AddedToRematList = Cands.updateOrInsert(R, DAG.LIS);
          if (AddedToRematList)
            RematDefToLiveInRegions[Def].push_back(I);
        }
      }
    }
  }
  Cands.resolveSameBlockUses(&DAG.MRI, DAG.LIS);
}

bool PreRARematStage::createRematPlan(bool Aggressive) {
  DenseMap<unsigned, unsigned> OptRegions;
  OptRegionRPReduction.clear();
  DenseMap<unsigned, GCNRPTracker::LiveRegSet> OptRegionLiveIns;
  for (unsigned I = 0, E = DAG.Regions.size(); I != E; ++I) {
    if (!RelevantRegions[I])
      continue;

    GCNRegPressure &RP = DAG.Pressure[I];
    if (ST.getOccupancyWithNumSGPRs(RP.getSGPRNum()) == DAG.MinOccupancy) {
      return false;
    }

    unsigned NumVGPRs = RP.getVGPRNum(ST.hasGFX90AInsts());
    unsigned Bias = Aggressive ? LiveInBias : LiveThruBias;
    int NumToIncreaseOcc = NumVGPRs + Bias - ST.getAddressableNumArchVGPRs();
    ST.getNumVGPRsToIncreaseOccupancy(NumVGPRs);

    OptRegionRPReduction[I] = NumToIncreaseOcc;
    OptRegionLiveIns[I] = DAG.LiveIns[I];
  }

  bool FoundAny = true;
  bool BadRP = true;
  unsigned Stage = 0;

  RematCandidates NewCandidates = Cands;
  while (BadRP && (FoundAny || (Stage < 3))) {
    if (FoundAny) {
      Stage = 0;
    } else {
      ++Stage;
    }
    FoundAny = false;
    RematCandidates RCCache = NewCandidates;
    NewCandidates.clear();
    RCCache.sort(DAG.LIS);

    for (const RematCandidate &R : reverse(RCCache.Sorted)) {
      bool ShouldRemat = false;
      for (unsigned HighRPRegion : R.HighRPRegions) {
        if (OptRegionRPReduction[HighRPRegion] > 0) {
          ShouldRemat = true;
          break;
        }
      }

      // should also check loop cycle depth
      if (!ShouldRemat && (Stage < 2)) {
        NewCandidates.insert(R);
        continue;
      }

      auto DefReg = R.Def->getOperand(0).getReg();
      LiveInterval &DefLI = DAG.LIS->getInterval(DefReg);
      LaneBitmask DefLanes = DAG.MRI.getMaxLaneMaskForVReg(DefReg);
      if (DefLI.hasSubRanges()) {
        unsigned SubReg = R.Def->getOperand(0).getSubReg();
        DefLanes =
            SubReg
                ? DAG.TRI->getSubRegIndexLaneMask(SubReg)
                : DAG.MRI.getMaxLaneMaskForVReg(R.Def->getOperand(0).getReg());
      }

      // for each of the uses, get a lanemask
      // check the impacted regions liveins for the use-mask
      // also check the lives for the def lanes
      // estimate the impact on RP, if we negatively impact RP in a good region,
      // defer the candidate.

      if (Stage < 1) {
        bool ShouldDefer = false;
        for (unsigned HighRPRegion : R.HighRPRegions) {
          if (OptRegionRPReduction[HighRPRegion] <= 0)
            continue;

          GCNRPTracker::LiveRegSet &TheLiveRegs =
              OptRegionLiveIns[HighRPRegion];
          int RPImpact = 0;
          if (TheLiveRegs.contains(DefReg)) {
            auto OldLiveIns = TheLiveRegs[DefReg];
            auto NonLiveIns = OldLiveIns & DefLanes;
            RPImpact += NonLiveIns.getNumLanes() / 2;
          }
          for (const MachineOperand &MO : R.Def->operands()) {
            if (RPImpact < 0)
              break;
            if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
              continue;
            auto Reg = MO.getReg();
            if (!Reg.isVirtual())
              continue;

            LiveInterval &UseLI = DAG.LIS->getInterval(MO.getReg());
            LaneBitmask UseLanes = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
            // Check that subrange is live at UseIdx.
            if (UseLI.hasSubRanges()) {
              unsigned SubReg = MO.getSubReg();
              UseLanes = SubReg ? DAG.TRI->getSubRegIndexLaneMask(SubReg)
                                : DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
            }

            if (!TheLiveRegs.contains(Reg)) {
              RPImpact -= UseLanes.getNumLanes() / 2;
              continue;
            }

            LaneBitmask OldLiveInMask = TheLiveRegs[Reg];
            if ((OldLiveInMask & UseLanes) == UseLanes)
              continue;
            LaneBitmask NewLanes = UseLanes & ~OldLiveInMask;
            RPImpact -= NewLanes.getNumLanes() / 2;
          }

          if (RPImpact < 0) {
            ShouldDefer = true;
            break;
          }
        }

        if (ShouldDefer) {
          NewCandidates.insert(R);
          continue;
        }
      }

      FoundAny = true;
      RematPlan.updateOrInsert(*const_cast<RematCandidate *>(&R), DAG.LIS);
      GCNRPTracker::LiveRegSet NewLiveIns;

      for (const MachineOperand &MO : R.Def->operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
          continue;
        auto Reg = MO.getReg();
        if (!Reg.isVirtual())
          continue;
        LiveInterval &UseLI = DAG.LIS->getInterval(MO.getReg());
        LaneBitmask CoveredLanes = DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
        // Check that subrange is live at UseIdx.
        if (UseLI.hasSubRanges()) {
          unsigned SubReg = MO.getSubReg();
          CoveredLanes = SubReg ? DAG.TRI->getSubRegIndexLaneMask(SubReg)
                                : DAG.MRI.getMaxLaneMaskForVReg(MO.getReg());
        }

        NewLiveIns[Reg] = CoveredLanes;

        if (!canRemat(Reg))
          continue;

        bool FoundInBlockUse = false;

        for (auto &TheUseInst : DAG.MRI.use_nodbg_instructions(Reg)) {
          for (auto HighRegion : R.HighRPRegions) {
            auto TheBlock = DAG.Regions[HighRegion].first->getParent();
            if (TheUseInst.getParent() == TheBlock) {
              FoundInBlockUse = true;
              break;
            }
          }
        }

        if (!Aggressive && FoundInBlockUse)
          continue;

        MachineInstr *UseDef = DAG.MRI.getOneDef(Reg)->getParent();
        MachineBasicBlock::iterator UseRematPt;
        if (R.InsertPt != R.InsertPt->getParent()->begin())
          UseRematPt = std::prev(R.InsertPt);
        else {
          UseRematPt = R.InsertPt->getParent()->begin();
        }
        RematCandidate RNew(UseDef, CI.getCycleDepth(UseRematPt->getParent()),
                            R.HighRPRegions, UseRematPt);
        NewCandidates.updateOrInsert(RNew, DAG.LIS);
      }

      for (unsigned HighRPRegion : R.HighRPRegions) {
        int RPImpact = 0;
        GCNRPTracker::LiveRegSet &TheLiveRegs = OptRegionLiveIns[HighRPRegion];

        unsigned DefReg = R.Def->getOperand(0).getReg();
        if (TheLiveRegs.contains(DefReg)) {
          auto OldLiveIns = TheLiveRegs[DefReg];
          auto NonLiveIns = OldLiveIns & DefLanes;
          RPImpact += NonLiveIns.getNumLanes() / 2;
          if (OldLiveIns == NonLiveIns) {
            TheLiveRegs[DefReg] = LaneBitmask(0);
            TheLiveRegs.erase(DefReg);
          } else {
            TheLiveRegs[DefReg] = OldLiveIns & ~NonLiveIns;
          }
        }

        for (auto &LiveInPair : NewLiveIns) {
          unsigned UseReg = LiveInPair.first;
          LaneBitmask UseMask = LiveInPair.second;
          if (!TheLiveRegs.contains(UseReg)) {
            RPImpact -= UseMask.getNumLanes() / 2;
            TheLiveRegs[UseReg] = UseMask;
            continue;
          }

          // An existing live in
          LaneBitmask OldLiveInMask = TheLiveRegs[UseReg];
          // Already fully covered
          if ((OldLiveInMask & UseMask) == UseMask)
            continue;
          LaneBitmask NewLanes = UseMask & ~OldLiveInMask;
          RPImpact -= NewLanes.getNumLanes() / 2;
          TheLiveRegs[UseReg] |= UseMask;
        }

        OptRegionRPReduction[HighRPRegion] -= RPImpact;
      }

      BadRP = false;
      for (auto HighRPRegion : OptRegionRPReduction) {
        if (HighRPRegion.second > 0) {
          BadRP = true;
          break;
        }
      }
      if (!BadRP)
        break;
    }

    if (BadRP)
      NewCandidates.resolveSameBlockUses(&DAG.MRI, DAG.LIS);
  }

  BadRP = false;
  for (auto ReductionNeeded : OptRegionRPReduction) {
    if (ReductionNeeded.second > 0) {
      BadRP = true;
      break;
    }
  }

  if (!BadRP)
    RematPlan.resolveSameBlockUses(&DAG.MRI, DAG.LIS);

  return !BadRP;
}

bool PreRARematStage::implementRematPlan(const TargetInstrInfo *TII,
                                         bool Aggressive) {
  // Temporary copies of cached variables we will be modifying and replacing if
  // sinking succeeds.
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      RegionsCache;

  RegionsCache.resize(DAG.Regions.size());
  RegionsCache = DAG.Regions;
  auto &Regions = DAG.Regions;

  DenseMap<MachineInstr *, MachineInstr *> InsertedMIToOldDef;
  LiveIntervals *LIS = DAG.LIS;

// TODO -- enable
// May result in bad insert points (e.g. before exec set code)
//  if (!Aggressive)
//    RematPlan.hoistToDominator(&PDT, CI, TargetBlock);

  Cands.resolveSameBlockUses(&DAG.MRI, DAG.LIS);
  RematPlan.sort(LIS);
  for (auto I = RematPlan.Sorted.rbegin(), E = RematPlan.Sorted.rend(); I != E;
       I++) {
    auto R = *I;
    MachineInstr *Def = R.Def;
    MachineBasicBlock::iterator InsertPos = R.InsertPt;

    Register Reg = Def->getOperand(0).getReg();

    SmallVector<MachineInstr *, 8> UserInst;

    /// TODO -- fail if we dont hoist all instructions
    PDT.recalculate(MF);
    for (auto &UseI : DAG.MRI.use_nodbg_instructions(Reg)) {
      if (PDT.dominates(InsertPos->getParent(), UseI.getParent())) {
        UserInst.push_back(&UseI);
      }
      if (UseI.getParent() == InsertPos->getParent()) {
        if (SlotIndex::isEarlierInstr(
                DAG.LIS->getInstructionIndex(UseI).getRegSlot(),
                DAG.LIS->getInstructionIndex(*InsertPos).getRegSlot())) {
                  InsertPos = MachineBasicBlock::iterator(&UseI);
                }
      }
    }

    for (auto &Op : Def->operands()) {
      if (!Op.isReg())
        continue;
      auto OpReg = Op.getReg();
      if (!OpReg.isVirtual())
        continue;

      for (auto &DefI : DAG.MRI.def_instructions(Reg)) {
        if (DefI.getParent() != InsertPos->getParent())
          continue;

        if (SlotIndex::isEarlierInstr(
                DAG.LIS->getInstructionIndex(DefI).getRegSlot(),
                DAG.LIS->getInstructionIndex(*InsertPos).getRegSlot()))
          continue;

        MachineBasicBlock::iterator DefIt = MachineBasicBlock::iterator(&DefI);
        InsertPos = std::next(DefIt);
      }
    }

    TII->reMaterialize(*InsertPos->getParent(), InsertPos, Reg,
                       Def->getOperand(0).getSubReg(), *Def);
    MachineInstr *NewMI = &*std::prev(InsertPos);
    NewMI->getOperand(0).setSubReg(Def->getOperand(0).getSubReg());
    NewMI->clearRegisterDeads(Def->getOperand(0).getReg());

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(Reg);
    Register NewReg = DAG.MRI.createVirtualRegister(RC);
    NewMI->getOperand(0).setReg(NewReg);
    auto X = DAG.MRI.use_nodbg_instructions(Reg);
    SmallVector<MachineInstr *, 4> Users;
    for (auto &UseI : X) {
      Users.push_back(&UseI);
    }
    for (auto UseI : UserInst) {
      for (MachineOperand &Op : UseI->operands()) {
        if (!Op.isReg())
          continue;
        Register UseReg = Op.getReg();
        if (UseReg == Reg)
          Op.setReg(NewReg);
      }
    }

    LIS->InsertMachineInstrInMaps(*NewMI);
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
    LIS->createAndComputeVirtRegInterval(NewReg);
    InsertedMIToOldDef[NewMI] = Def;

    for (const MachineOperand &MO : NewMI->operands()) {
      if (!MO.isReg() || !MO.getReg() || !MO.readsReg())
        continue;
      auto UseReg = MO.getReg();
      if (!UseReg.isVirtual())
        continue;

      LIS->removeInterval(UseReg);
      LIS->createAndComputeVirtRegInterval(UseReg);
    }

    // Update region boundaries in scheduling region we sinked from since we
    // may sink an instruction that was at the beginning or end of its region

    DAG.updateRegionBoundaries(DAG.Regions, Def, nullptr);
    // Update region boundaries in region we sinked to.
    DAG.updateRegionBoundaries(DAG.Regions, InsertPos, NewMI);
  }

  auto NewLiveIns = DAG.getRegionLiveInMap();

  for (unsigned K = 0; K < Regions.size(); K++) {
    DAG.LiveIns[K] = NewLiveIns[&*DAG.Regions[K].first];
  }

  RescheduleRegions.set();

  if (GCNTrackers)
    DAG.RegionLiveOuts.buildLiveRegMap();

  SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  MFI.increaseOccupancy(MF, ++DAG.MinOccupancy);
  return true;
}

// Copied from MachineLICM
bool PreRARematStage::isTriviallyReMaterializable(const MachineInstr &MI) {
  if (MI.getNumDefs() > 1 || !DAG.TII->isTriviallyReMaterializable(MI))
    return false;

  return true;
}

// When removing, we will have to check both beginning and ending of the region.
// When inserting, we will only have to check if we are inserting NewMI in front
// of a scheduling region and do not need to check the ending since we will only
// ever be inserting before an already existing MI.
void GCNScheduleDAGMILive::updateRegionBoundaries(
    SmallVectorImpl<std::pair<MachineBasicBlock::iterator,
                              MachineBasicBlock::iterator>> &RegionBoundaries,
    MachineBasicBlock::iterator MI, MachineInstr *NewMI) {
  unsigned I = 0, E = RegionBoundaries.size();
  // Search for first region of the block where MI is located. We may encounter
  // an empty region if all instructions from an initially non-empty region were
  // removed.
  while (I != E && RegionBoundaries[I].first != RegionBoundaries[I].second &&
         MI->getParent() != RegionBoundaries[I].first->getParent())
    ++I;

  for (; I != E; ++I) {
    auto &Bounds = RegionBoundaries[I];
    // assert(MI != Bounds.second && "cannot insert at region end");
    // assert(!NewMI || NewMI != Bounds.second && "cannot remove at region
    // end");

    // We may encounter an empty region if all of the region' instructions were
    // previously removed.
    if (Bounds.first == Bounds.second) {
      if (MI->getParent()->end() != Bounds.second)
        return;
      continue;
    }
    if (MI->getParent() != Bounds.first->getParent())
      return;

    // We only care for modifications at the beginning of the region since the
    // upper region boundary is exclusive.
    if (MI != Bounds.first)
      continue;
    if (!NewMI) {
      // This is an MI removal, which may leave the region empty; in such cases
      // set both boundaries to the removed instruction's MBB's end.
      MachineBasicBlock::iterator NextMI = std::next(MI);
      if (NextMI != Bounds.second)
        Bounds.first = NextMI;
      else
        Bounds.first = Bounds.second;
    } else {
      // This is an MI insertion at the beggining of the region.
      Bounds.first = NewMI;
    }
    return;
  }
}

static bool hasIGLPInstrs(ScheduleDAGInstrs *DAG) {
  return any_of(*DAG, [](MachineBasicBlock::iterator MI) {
    unsigned Opc = MI->getOpcode();
    return Opc == AMDGPU::SCHED_GROUP_BARRIER || Opc == AMDGPU::IGLP_OPT;
  });
}

GCNPostSchedStrategy::GCNPostSchedStrategy(const MachineSchedContext *C)
    : PostGenericScheduler(C) {}

static void tracePick(GenericSchedulerBase::CandReason Reason, bool IsTop) {
  LLVM_DEBUG(dbgs() << "Pick " << (IsTop ? "Top " : "Bot ")
                    << GenericSchedulerBase::getReasonStr(Reason) << '\n');
}

static void tracePick(const GenericSchedulerBase::SchedCandidate &Cand) {
  tracePick(Cand.Reason, Cand.AtTop);
}

SUnit *GCNPostSchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  do {
    if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (SU) {
        tracePick(Only1, true);
      } else {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        // Set the bottom-up policy based on the state of the current bottom
        // zone and the instructions outside the zone, including the top zone.
        setPolicy(BotCand.Policy, /*IsPostRA=*/true, Bot, nullptr);
        pickNodeFromQueue(Bot, BotCand);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        tracePick(BotCand);
        SU = BotCand.SU;
      }
      IsTopNode = false;
    } else if (RegionPolicy.OnlyTopDown) {
      SU = Top.pickOnlyChoice();
      if (SU) {
        tracePick(Only1, true);
      } else {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        // Set the top-down policy based on the state of the current top zone
        // and the instructions outside the zone, including the bottom zone.
        setPolicy(TopCand.Policy, /*IsPostRA=*/true, Top, nullptr);
        pickNodeFromQueue(Top, TopCand);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        tracePick(TopCand);
        SU = TopCand.SU;
      }
      IsTopNode = true;
    } else {
      SU = pickNodeBidirectional(IsTopNode);
    }
  } while (SU->isScheduled);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  if (CustomResTracking) {
#ifndef NDEBUG
    unsigned XDLCyclesBefore = XDLProcRes.CyclesReserved;
#endif

    const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(DAG->TII);
    bool IsXDL = TII->isXDL(*SU->getInstr());
    unsigned Cycles = SU->Latency;
    if (IsXDL) {
      // FIXME: Hack since XDL is only actually occupying for 24 cycles with 8
      // pass MFMA.
      if (Cycles > 2)
        Cycles -= 2;
      XDLProcRes.reset();
      XDLProcRes.reserve(Cycles);
    } else {
      XDLProcRes.release(Cycles);
    }

    LLVM_DEBUG(dbgs() << "OldXDLProcRes: " << XDLCyclesBefore
                      << "\nNewXDLProcRes: " << XDLProcRes.CyclesReserved
                      << "\n");
  }

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

bool GCNPostSchedStrategy::tryXDL(SchedCandidate &Cand,
                                  SchedCandidate &TryCand) {
  MachineInstr *CInst = Cand.SU->getInstr();
  MachineInstr *TCInst = TryCand.SU->getInstr();
  const SIInstrInfo *TII = DAG->MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  bool CandIsXDL = TII->isXDL(*CInst);
  bool TryCandIsXDL = TII->isXDL(*TCInst);

  LLVM_DEBUG(dbgs() << "tryXDL"; DAG->dumpNode(*Cand.SU);
             DAG->dumpNode(*TryCand.SU));

  // XDL is free.
  if (XDLProcRes.CyclesReserved == 0) {
    if (!TryCandIsXDL && !CandIsXDL)
      return false;

    if (TryCandIsXDL && !CandIsXDL) {
      TryCand.Reason = ResourceDemand;
      return true;
    }

    if (CandIsXDL && !TryCandIsXDL) {
      Cand.Reason = ResourceDemand;
      return true;
    }

    // XDL is free and both candidates are XDL.
    if (CandIsXDL && TryCandIsXDL) {
      unsigned CandReadyVALUSuccs = 0;
      unsigned TryReadyVALUSuccs = 0;

      // Count VALU successors that would become ready after scheduling this
      // MFMA.
      SmallPtrSet<SUnit *, 8> CandSeenSuccs;
      for (SDep &Succ : Cand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!CandSeenSuccs.insert(SuccSU).second)
          continue;

        if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
          ++CandReadyVALUSuccs;
        }
      }

      SmallPtrSet<SUnit *, 8> TrySeenSuccs;
      for (SDep &Succ : TryCand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!TrySeenSuccs.insert(SuccSU).second)
          continue;

        if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
          ++TryReadyVALUSuccs;
        }
      }

      LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                        << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

      // Prefer the candidate that would immediately free up more VALU
      // instructions
      if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
        Cand.Reason = ResourceDemand;
        return true;
      }
      if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
        TryCand.Reason = ResourceDemand;
        return true;
      }
      if (CandReadyVALUSuccs > 0) {
        Cand.Reason = ResourceDemand;
        return true;
      }

      // If they free up the same number of VALUs, fall back to the old
      // heuristic and just count the total number of VALU successors
      unsigned CandVALUSuccs = 0;
      unsigned TryVALUSuccs = 0;

      // Reset and reuse our sets of seen successors
      CandSeenSuccs.clear();
      TrySeenSuccs.clear();

      for (SDep &Succ : Cand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!CandSeenSuccs.insert(SuccSU).second)
          continue;
        if (TII->isVALU(*SuccSU->getInstr()))
          ++CandVALUSuccs;
      }

      for (SDep &Succ : TryCand.SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!TrySeenSuccs.insert(SuccSU).second)
          continue;
        if (TII->isVALU(*SuccSU->getInstr()))
          ++TryVALUSuccs;
      }

      LLVM_DEBUG(dbgs() << "CandVALUSuccs: " << CandVALUSuccs
                        << " TryVALUSuccs: " << TryVALUSuccs << "\n");

      // If one candidate has more total VALU successors, prefer it.
      if (CandVALUSuccs > TryVALUSuccs) {
        Cand.Reason = ResourceDemand;
        return true;
      }
      if (CandVALUSuccs < TryVALUSuccs) {
        TryCand.Reason = ResourceDemand;
        return true;
      }

      Cand.Reason = ResourceDemand;
      return true;
    }
  }

  assert(XDLProcRes.CyclesReserved);

  // XDL is in use and Cand is a MFMA.
  if (CandIsXDL) {
    if (!TryCandIsXDL) {
      TryCand.Reason = ResourceReduce;
      return true;
    }
    // TryCandIsXDL and CandIsXDL and resource is in use.
    unsigned CandReadyVALUSuccs = 0;
    unsigned TryReadyVALUSuccs = 0;

    // Count VALU successors that would become ready after scheduling this
    // MFMA.
    SmallPtrSet<SUnit *, 8> CandSeenSuccs;
    for (SDep &Succ : Cand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!CandSeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++CandReadyVALUSuccs;
      }
    }

    SmallPtrSet<SUnit *, 8> TrySeenSuccs;
    for (SDep &Succ : TryCand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!TrySeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++TryReadyVALUSuccs;
      }
    }

    LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                      << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

    // Prefer the candidate that would immediately free up more VALU
    // instructions
    if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
      TryCand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs > 0) {
      Cand.Reason = ResourceDemand;
      return true;
    }

    // Let other heuristics take precedence if both are MFMA and resource is
    // in use.
    return false;
  }

  // Both candidates are not MFMA and resource is in use.
  if (!TryCandIsXDL) {
    unsigned CandReadyVALUSuccs = 0;
    unsigned TryReadyVALUSuccs = 0;

    SmallPtrSet<SUnit *, 8> CandSeenSuccs;
    for (SDep &Succ : Cand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!CandSeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++CandReadyVALUSuccs;
      }
    }

    SmallPtrSet<SUnit *, 8> TrySeenSuccs;
    for (SDep &Succ : TryCand.SU->Succs) {
      SUnit *SuccSU = Succ.getSUnit();
      if (!TrySeenSuccs.insert(SuccSU).second)
        continue;

      if (TII->isVALU(*SuccSU->getInstr()) && SuccSU->NumPredsLeft == 1) {
        ++TryReadyVALUSuccs;
      }
    }

    LLVM_DEBUG(dbgs() << "CandReadyVALUSuccs: " << CandReadyVALUSuccs
                      << " TryReadyVALUSuccs: " << TryReadyVALUSuccs << "\n");

    // Prefer the candidate that would immediately free up more VALU
    // instructions
    if (CandReadyVALUSuccs > TryReadyVALUSuccs) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs < TryReadyVALUSuccs) {
      TryCand.Reason = ResourceDemand;
      return true;
    }
    if (CandReadyVALUSuccs > 0) {
      Cand.Reason = ResourceDemand;
      return true;
    }
    unsigned CandCycles = Cand.SU->Latency;
    unsigned TryCandCycles = TryCand.SU->Latency;

    // Check if either instruction would cause XDL resources to go negative
    bool CandOverflow = CandCycles > XDLProcRes.CyclesReserved;
    bool TryCandOverflow = TryCandCycles > XDLProcRes.CyclesReserved;

    // If one overflows and the other doesn't, prefer the one that overflows
    // because it will free up the XDL resource
    if (CandOverflow && !TryCandOverflow) {
      Cand.Reason = ResourceReduce;
      return true;
    }
    if (TryCandOverflow && !CandOverflow) {
      TryCand.Reason = ResourceReduce;
      return true;
    }

    // Both would overflow or neither would - pick the one that gets closest to
    // zero
    int CandRemainingXDL = static_cast<int>(XDLProcRes.CyclesReserved) -
                           static_cast<int>(CandCycles);
    int TryCandRemainingXDL = static_cast<int>(XDLProcRes.CyclesReserved) -
                              static_cast<int>(TryCandCycles);

    if (std::abs(TryCandRemainingXDL) < std::abs(CandRemainingXDL)) {
      TryCand.Reason = ResourceReduce;
      return true;
    }
    Cand.Reason = ResourceReduce;
    return true;
  }

  // XDL resource is in use and Cand is not MFMA but TryCand is.
  Cand.Reason = ResourceReduce;
  return true;
}

bool GCNPostSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                        SchedCandidate &TryCand) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  if (CustomResTracking && tryXDL(Cand, TryCand))
    return TryCand.Reason != NoCand;

  // Prioritize instructions that read unbuffered resources by stall cycles.
  if (tryLess(Top.getLatencyStallCycles(TryCand.SU),
              Top.getLatencyStallCycles(Cand.SU), TryCand, Cand, Stall))
    return TryCand.Reason != NoCand;

  // Keep clustered nodes together.
  unsigned CandZoneCluster = Cand.AtTop ? TopClusterID : BotClusterID;
  unsigned TryCandZoneCluster = TryCand.AtTop ? TopClusterID : BotClusterID;
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;

  // Avoid critical resource consumption and balance the schedule.
  if (tryLess(TryCand.ResDelta.CritResources, Cand.ResDelta.CritResources,
              TryCand, Cand, ResourceReduce))
    return TryCand.Reason != NoCand;
  if (tryGreater(TryCand.ResDelta.DemandedResources,
                 Cand.ResDelta.DemandedResources, TryCand, Cand,
                 ResourceDemand))
    return TryCand.Reason != NoCand;

  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary.
  if (Cand.AtTop == TryCand.AtTop) {
    // Avoid serializing long latency dependence chains.
    if (Cand.Policy.ReduceLatency &&
        tryLatency(TryCand, Cand, Cand.AtTop ? Top : Bot))
      return TryCand.Reason != NoCand;
  }

  // Fall through to original instruction order.
  if (TryCand.SU->NodeNum < Cand.SU->NodeNum) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  return false;
}

GCNPostScheduleDAGMILive::GCNPostScheduleDAGMILive(
    MachineSchedContext *C, std::unique_ptr<GCNPostSchedStrategy> S,
    bool RemoveKillFlags)
    : ScheduleDAGMI(C, std::move(S), RemoveKillFlags) {}

void GCNPostScheduleDAGMILive::schedule() {
  HasIGLPInstrs = hasIGLPInstrs(this);
  S = static_cast<GCNPostSchedStrategy *>(SchedImpl.get());
  S->CustomResTracking = HasIGLPInstrs;

  if (HasIGLPInstrs) {
    SavedMutations.clear();
    SavedMutations.swap(Mutations);
    addMutation(createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::PostRA));
  }

  ScheduleDAGMI::schedule();
}

void GCNPostScheduleDAGMILive::finalizeSchedule() {
  if (HasIGLPInstrs)
    SavedMutations.swap(Mutations);

  ScheduleDAGMI::finalizeSchedule();
}
