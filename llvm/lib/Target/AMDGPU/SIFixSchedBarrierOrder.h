#ifndef LLVM_LIB_TARGET_AMDGPU_SIFIXSCHEDBARRIERORDER_H
#define LLVM_LIB_TARGET_AMDGPU_SIFIXSCHEDBARRIERORDER_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class SIFixSchedBarrierOrderPass
    : public PassInfoMixin<SIFixSchedBarrierOrderPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif
