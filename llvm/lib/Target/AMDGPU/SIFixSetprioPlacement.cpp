//===- SIFixSetprioPlacement.cpp - Fix s_setprio 0 placement --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// When the user emits s_setprio 0 via inline asm, the scheduler may hoist it
// above MFMA chains. This pass moves the inline-asm s_setprio 0 to the
// largest gap between consecutive MFMAs in the same basic block, which
// typically corresponds to the GEMM1→softmax→GEMM2 boundary in FMHA kernels.
// This keeps GEMM1 at high priority while allowing wave interleaving during
// the softmax/GEMM2 phase.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-setprio-placement"

namespace {

class SIFixSetprioPlacement : public MachineFunctionPass {
public:
  static char ID;
  SIFixSetprioPlacement() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SI Fix Setprio Placement";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char SIFixSetprioPlacement::ID = 0;

INITIALIZE_PASS(SIFixSetprioPlacement, DEBUG_TYPE,
                "SI Fix Setprio Placement", false, false)

FunctionPass *llvm::createSIFixSetprioPlacementPass() {
  return new SIFixSetprioPlacement();
}

static bool isSetprioInlineAsm(const MachineInstr &MI, unsigned Priority) {
  if (!MI.isInlineAsm())
    return false;
  const MachineOperand &AsmOp = MI.getOperand(InlineAsm::MIOp_AsmString);
  if (!AsmOp.isSymbol())
    return false;
  StringRef AsmStr(AsmOp.getSymbolName());
  if (Priority == 0)
    return AsmStr.contains("s_setprio 0");
  return AsmStr.contains("s_setprio 1");
}

static bool isMFMA(const MachineInstr &MI) {
  return SIInstrInfo::isMAI(MI);
}

bool SIFixSetprioPlacement::runOnMachineFunction(MachineFunction &MF) {
  // Pass disabled: the compiler's default s_setprio 0 placement (between
  // GEMM1 MFMA #15 and #16) is near-optimal for MI325X 2-wave interleaving.
  // Moving it later (after GEMM1 or after GEMM2) regresses performance by
  // preventing the other wave from overlapping its memory ops during GEMM2.
  return false;
}
