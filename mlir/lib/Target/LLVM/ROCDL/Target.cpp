//===- Target.cpp - MLIR LLVM ROCDL target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related functions including registration
// calls for the `#rocdl.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ROCDL/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/TargetParser.h"

#include <cstdlib>
#include <optional>

using namespace mlir;
using namespace mlir::ROCDL;

#ifndef __DEFAULT_ROCM_PATH__
#define __DEFAULT_ROCM_PATH__ ""
#endif

namespace {
// Implementation of the `TargetAttrInterface` model.
class ROCDLTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<ROCDLTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the ROCDL dialect, the ROCDL translation and the target interface.
void mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
    ROCDLTargetAttr::attachInterface<ROCDLTargetAttrImpl>(*ctx);
  });
}

void mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

// Search for the ROCM path.
StringRef mlir::ROCDL::getROCMPath() {
  if (const char *var = std::getenv("ROCM_PATH"))
    return var;
  if (const char *var = std::getenv("ROCM_ROOT"))
    return var;
  if (const char *var = std::getenv("ROCM_HOME"))
    return var;
  return __DEFAULT_ROCM_PATH__;
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, ROCDLTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(),
                     target.getFeatures(), target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()),
      librariesToLink(targetOptions.getLibrariesToLink()) {

  // If `targetOptions` has an empty toolkitPath use `getROCMPath`
  if (toolkitPath.empty())
    toolkitPath = getROCMPath();

  // Append the files in the target attribute.
  if (target.getLink())
    librariesToLink.append(target.getLink().begin(), target.getLink().end());
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
  // If the `AMDGPU` LLVM target was built, initialize it.
#if MLIR_ENABLE_ROCM_CONVERSIONS
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
#endif
  });
}

ROCDLTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

ArrayRef<Attribute> SerializeGPUModuleBase::getLibrariesToLink() const {
  return librariesToLink;
}

LogicalResult SerializeGPUModuleBase::appendStandardLibs(AMDGCNLibraries libs) {
  if (libs == AMDGCNLibraries::None)
    return success();
  StringRef pathRef = getToolkitPath();

  // Get the path for the device libraries
  SmallString<256> path;
  path.insert(path.begin(), pathRef.begin(), pathRef.end());
  llvm::sys::path::append(path, "amdgcn", "bitcode");
  pathRef = StringRef(path.data(), path.size());

  // Fail if the path is invalid.
  if (!llvm::sys::fs::is_directory(pathRef)) {
    getOperation().emitError() << "ROCm amdgcn bitcode path: " << pathRef
                               << " does not exist or is not a directory";
    return failure();
  }

  // Helper function for adding a library.
  auto addLib = [&](const Twine &lib) -> bool {
    auto baseSize = path.size();
    llvm::sys::path::append(path, lib);
    StringRef pathRef(path.data(), path.size());
    if (!llvm::sys::fs::is_regular_file(pathRef)) {
      getOperation().emitRemark() << "bitcode library path: " << pathRef
                                  << " does not exist or is not a file";
      return true;
    }
    librariesToLink.push_back(StringAttr::get(target.getContext(), pathRef));
    path.truncate(baseSize);
    return false;
  };

  // Add ROCm device libraries. Fail if any of the libraries is not found, ie.
  // if any of the `addLib` failed.
  if ((any(libs & AMDGCNLibraries::Ocml) && addLib("ocml.bc")) ||
      (any(libs & AMDGCNLibraries::Ockl) && addLib("ockl.bc")) ||
      (any(libs & AMDGCNLibraries::Hip) && addLib("hip.bc")) ||
      (any(libs & AMDGCNLibraries::OpenCL) && addLib("opencl.bc")))
    return failure();
  return success();
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module) {
  // Return if there are no libs to load.
  if (deviceLibs == AMDGCNLibraries::None && librariesToLink.empty())
    return SmallVector<std::unique_ptr<llvm::Module>>();
  if (failed(appendStandardLibs(deviceLibs)))
    return std::nullopt;
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), librariesToLink,
                                      bcFiles, true)))
    return std::nullopt;
  return std::move(bcFiles);
}

LogicalResult SerializeGPUModuleBase::handleBitcodeFile(llvm::Module &module) {
  // Some ROCM builds don't strip this like they should
  if (auto *openclVersion = module.getNamedMetadata("opencl.ocl.version"))
    module.eraseNamedMetadata(openclVersion);
  // Stop spamming us with clang version numbers
  if (auto *ident = module.getNamedMetadata("llvm.ident"))
    module.eraseNamedMetadata(ident);
  // Override the libModules datalayout and target triple with the compiler's
  // data layout should there be a discrepency.
  setDataLayoutAndTriple(module);
  return success();
}

void SerializeGPUModuleBase::handleModulePreLink(llvm::Module &module) {
  // If all libraries are not set, traverse the module to determine which
  // libraries are required.
  if (deviceLibs != AMDGCNLibraries::All) {
    for (llvm::Function &f : module.functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        StringRef funcName = f.getName();
        if ("printf" == funcName)
          deviceLibs |= AMDGCNLibraries::OpenCL | AMDGCNLibraries::Ockl |
                        AMDGCNLibraries::Ocml;
        if (funcName.starts_with("__ockl_"))
          deviceLibs |= AMDGCNLibraries::Ockl;
        if (funcName.starts_with("__ocml_"))
          deviceLibs |= AMDGCNLibraries::Ocml;
        if (funcName == "__atomic_work_item_fence")
          deviceLibs |= AMDGCNLibraries::Hip;
      }
    }
  }
  addControlVariables(module, deviceLibs, target.hasWave64(), target.hasDaz(),
                      target.hasFiniteOnly(), target.hasUnsafeMath(),
                      target.hasFastMath(), target.hasCorrectSqrt(),
                      target.getAbi());
}

void SerializeGPUModuleBase::addControlVariables(
    llvm::Module &module, AMDGCNLibraries libs, bool wave64, bool daz,
    bool finiteOnly, bool unsafeMath, bool fastMath, bool correctSqrt,
    StringRef abiVer) {
  // Helper function for adding control variables.
  auto addControlVariable = [&module](StringRef name, uint32_t value,
                                      uint32_t bitwidth) {
    if (module.getNamedGlobal(name))
      return;
    llvm::IntegerType *type =
        llvm::IntegerType::getIntNTy(module.getContext(), bitwidth);
    llvm::GlobalVariable *controlVariable = new llvm::GlobalVariable(
        module, /*isConstant=*/type, true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(type, value), name, /*before=*/nullptr,
        /*threadLocalMode=*/llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
        /*addressSpace=*/4);
    controlVariable->setVisibility(
        llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    controlVariable->setAlignment(llvm::MaybeAlign(bitwidth / 8));
    controlVariable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  };

  // Note that COV6 requires ROCm 6.3+.
  int abi = 600;
  abiVer.getAsInteger(0, abi);
  module.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version", abi);
  // Return if no device libraries are required.
  if (libs == AMDGCNLibraries::None)
    return;
  // Add ocml related control variables.
  if (any(libs & AMDGCNLibraries::Ocml)) {
    addControlVariable("__oclc_finite_only_opt", finiteOnly || fastMath, 8);
    addControlVariable("__oclc_daz_opt", daz || fastMath, 8);
    addControlVariable("__oclc_correctly_rounded_sqrt32",
                       correctSqrt && !fastMath, 8);
    addControlVariable("__oclc_unsafe_math_opt", unsafeMath || fastMath, 8);
  }
  // Add ocml or ockl related control variables.
  if (any(libs & (AMDGCNLibraries::Ocml | AMDGCNLibraries::Ockl))) {
    addControlVariable("__oclc_wavefrontsize64", wave64, 8);
    // Get the ISA version.
    llvm::AMDGPU::IsaVersion isaVersion = llvm::AMDGPU::getIsaVersion(chip);
    // Add the ISA control variable.
    addControlVariable("__oclc_ISA_version",
                       isaVersion.Minor + 100 * isaVersion.Stepping +
                           1000 * isaVersion.Major,
                       32);
    addControlVariable("__oclc_ABI_version", abi, 32);
  }
}

std::optional<SmallVector<char, 0>>
SerializeGPUModuleBase::assembleIsa(StringRef isa) {
  auto loc = getOperation().getLoc();

  StringRef targetTriple = this->triple;

  SmallVector<char, 0> result;
  llvm::raw_svector_ostream os(result);

  llvm::Triple triple(llvm::Triple::normalize(targetTriple));
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return std::nullopt;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa), SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(targetTriple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, targetTriple, mcOptions));
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(targetTriple, chip, features));

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(target->createMCObjectStreamer(
      triple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti));

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    emitError(loc, "assembler initialization error");
    return std::nullopt;
  }

  parser->setTargetParser(*tap);
  bool parseError = parser->Run(false);
  if (parseError)
    llvm::errs() << "[assembleIsa] MC parser reported errors! result size="
                 << result.size() << "\n";
  else
    llvm::errs() << "[assembleIsa] OK, result size=" << result.size() << "\n";
  return std::move(result);
}

std::optional<SmallVector<char, 0>>
SerializeGPUModuleBase::compileToBinary(const std::string &serializedISA) {
  // Debug: dump ISA text entering the assembler
  {
    static int cbtDumpIdx = 0;
    std::string path =
        "/tmp/compile_to_binary_" + std::to_string(cbtDumpIdx++) + ".s";
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (!ec)
      os << serializedISA;
    llvm::errs() << "[compileToBinary] dumped " << serializedISA.size()
                 << " bytes to " << path << "\n";
  }
  // Assemble the ISA.
  std::optional<SmallVector<char, 0>> isaBinary = assembleIsa(serializedISA);

  if (!isaBinary) {
    getOperation().emitError() << "failed during ISA assembling";
    return std::nullopt;
  }

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    getOperation().emitError()
        << "failed to create a temporary file for dumping the ISA binary";
    return std::nullopt;
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  {
    llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
    tempIsaBinaryOs << StringRef(isaBinary->data(), isaBinary->size());
    tempIsaBinaryOs.flush();
  }

  // Create a temp file for HSA code object.
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    getOperation().emitError()
        << "failed to create a temporary file for the HSA code object";
    return std::nullopt;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  llvm::SmallString<128> lldPath(toolkitPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath,
      {"ld.lld", "-shared", tempIsaBinaryFilename, "-o", tempHsacoFilename});
  if (lldResult != 0) {
    getOperation().emitError() << "lld invocation failed";
    return std::nullopt;
  }

  // Load the HSA code object.
  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    getOperation().emitError()
        << "failed to read the HSA code object from the temp file";
    return std::nullopt;
  }

  StringRef buffer = (*hsacoFile)->getBuffer();

  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

static std::string processOneLoop(std::string loop,
                                   const std::string &label,
                                   bool insertYield = true,
                                   bool fillHazardGap = false,
                                   bool hoistVcmp = false) {
  // --- Pass 1: Move v_cmp_lt_i32_e64 before first barrier ---
  size_t firstBarrier = loop.find("s_barrier");
  if (firstBarrier == std::string::npos)
    return loop;

  // Find the waitcnt line before the first barrier (insertion point)
  size_t barrierLineStart = loop.rfind('\n', firstBarrier);
  if (barrierLineStart == std::string::npos) barrierLineStart = 0;
  else barrierLineStart++;

  // Look for the s_waitcnt before the barrier
  size_t insertPoint = barrierLineStart;
  {
    size_t lookBack = (barrierLineStart > 200) ? barrierLineStart - 200 : 0;
    std::string beforeBarrier = loop.substr(lookBack, barrierLineStart - lookBack);
    size_t lastWait = beforeBarrier.rfind("s_waitcnt");
    if (lastWait != std::string::npos) {
      size_t waitLineStart = beforeBarrier.rfind('\n', lastWait);
      insertPoint = lookBack + ((waitLineStart == std::string::npos) ? 0 : waitLineStart + 1);
    }
  }

  std::vector<std::string> cmpLines;
  size_t searchPos = firstBarrier;
  std::vector<std::pair<size_t, size_t>> linesToRemove;

  // Limit v_cmp hoisting to between the first and second barrier.
  // With 2x unrolled loops, Body B's v_cmps would overwrite Body A's SGPRs
  // if hoisted before the first barrier, corrupting mask predicates.
  size_t secondBarrier = loop.find("s_barrier", firstBarrier + 10);
  size_t vcmpSearchEnd = (secondBarrier != std::string::npos) ? secondBarrier : loop.size();

  while (hoistVcmp) {
    size_t pos = loop.find("v_cmp_lt_i32_e64 s[", searchPos);
    if (pos == std::string::npos || pos >= vcmpSearchEnd)
      break;
    size_t lineStart = loop.rfind('\n', pos);
    if (lineStart == std::string::npos)
      lineStart = 0;
    else
      lineStart++;
    size_t lineEnd = loop.find('\n', pos);
    if (lineEnd == std::string::npos)
      lineEnd = loop.size();
    else
      lineEnd++;

    std::string line = loop.substr(lineStart, lineEnd - lineStart);
    if (line.find("s[4:5]") == std::string::npos &&
        line.find("s[0:1]") == std::string::npos) {
      cmpLines.push_back(line);
      linesToRemove.push_back({lineStart, lineEnd});
    }
    searchPos = lineEnd;
  }

  // Partial hoisting: keep the LAST keepNearGemm2 v_cmps in place (near GEMM2)
  // to provide useful work that delays GEMM2 MFMA issue, reducing contention.
  const int keepNearGemm2 = 4;
  if ((int)cmpLines.size() > keepNearGemm2) {
    cmpLines.erase(cmpLines.end() - keepNearGemm2, cmpLines.end());
    linesToRemove.erase(linesToRemove.end() - keepNearGemm2, linesToRemove.end());
  } else {
    cmpLines.clear();
    linesToRemove.clear();
  }

  unsigned cmpMoved = cmpLines.size();
  unsigned nopFilled = 0;
  if (!cmpLines.empty()) {
    for (int i = linesToRemove.size() - 1; i >= 0; i--)
      loop.erase(linesToRemove[i].first,
                 linesToRemove[i].second - linesToRemove[i].first);

    // Recalculate insert point after erasures
    firstBarrier = loop.find("s_barrier");
    if (firstBarrier == std::string::npos)
      return loop;
    barrierLineStart = loop.rfind('\n', firstBarrier);
    if (barrierLineStart == std::string::npos) barrierLineStart = 0;
    else barrierLineStart++;
    insertPoint = barrierLineStart;
    {
      size_t lookBack = (barrierLineStart > 200) ? barrierLineStart - 200 : 0;
      std::string beforeBarrier = loop.substr(lookBack, barrierLineStart - lookBack);
      size_t lastWait = beforeBarrier.rfind("s_waitcnt");
      if (lastWait != std::string::npos) {
        size_t waitLineStart = beforeBarrier.rfind('\n', lastWait);
        insertPoint = lookBack + ((waitLineStart == std::string::npos) ? 0 : waitLineStart + 1);
      }
    }

    std::string insertBlock;
    for (const auto &line : cmpLines)
      insertBlock += line;
    loop.insert(insertPoint, insertBlock);
  }

  // --- Pass 2: Reserved for future scheduling optimizations ---
  unsigned yieldInserted = 0;

  // Detect 2x unrolled loops by counting barriers (>2 means multi-body).
  // Skip aggressive waitcnt optimizations for multi-body loops to avoid
  // incorrectly removing waits across body boundaries.
  unsigned barrierCount = 0;
  {
    size_t bpos = 0;
    while (true) {
      size_t b = loop.find("s_barrier", bpos);
      if (b == std::string::npos) break;
      barrierCount++;
      bpos = b + 10;
    }
  }
  bool isMultiBody = barrierCount > 2;

  // --- Pass 3: Remove redundant s_waitcnt vmcnt(4) ---
  if (!isMultiBody) {
    size_t pos = 0;
    while (true) {
      size_t first = loop.find("s_waitcnt vmcnt(4)", pos);
      if (first == std::string::npos) break;
      size_t firstEnd = loop.find('\n', first);
      if (firstEnd == std::string::npos) break;
      firstEnd++;
      size_t second = loop.find("s_waitcnt vmcnt(4)", firstEnd);
      if (second == std::string::npos) break;
      std::string between = loop.substr(firstEnd, second - firstEnd);
      if (between.find("buffer_load") == std::string::npos &&
          between.find("global_load") == std::string::npos) {
        size_t secLineStart = loop.rfind('\n', second);
        if (secLineStart == std::string::npos) secLineStart = 0;
        else secLineStart++;
        size_t secLineEnd = loop.find('\n', second);
        if (secLineEnd != std::string::npos) secLineEnd++;
        else secLineEnd = loop.size();
        loop.erase(secLineStart, secLineEnd - secLineStart);
        llvm::errs() << "[postProcessISA] Removed redundant vmcnt(4) in "
                     << label << "\n";
      }
      pos = firstEnd;
    }
  }

  unsigned vmcntRelaxed = 0;

  // --- Pass 4a: Remove redundant vmcnt(0) in V staging ---
  // DISABLED: the extra wait acts as a beneficial scheduling fence.
  if (false) {
    size_t first = loop.find("s_waitcnt vmcnt(0)\n");
    if (first == std::string::npos)
      first = loop.find("s_waitcnt vmcnt(0)");
    if (first != std::string::npos) {
      size_t firstEnd = loop.find('\n', first);
      if (firstEnd != std::string::npos) {
        firstEnd++;
        // Find next standalone vmcnt(0) (not part of a FULL wait)
        size_t pos2 = firstEnd;
        while (true) {
          size_t second = loop.find("s_waitcnt vmcnt(0)", pos2);
          if (second == std::string::npos) break;
          // Make sure it's standalone vmcnt(0), not part of FULL wait
          std::string restOfLine = loop.substr(second, 60);
          if (restOfLine.find("expcnt") != std::string::npos) {
            pos2 = second + 20;
            continue;
          }
          // Check no VMEM ops between first and second
          std::string between = loop.substr(firstEnd, second - firstEnd);
          if (between.find("buffer_load") == std::string::npos &&
              between.find("global_load") == std::string::npos) {
            size_t secLineStart = loop.rfind('\n', second);
            secLineStart = (secLineStart == std::string::npos) ? 0 : secLineStart + 1;
            size_t secLineEnd = loop.find('\n', second);
            secLineEnd = (secLineEnd == std::string::npos) ? loop.size() : secLineEnd + 1;
            loop.erase(secLineStart, secLineEnd - secLineStart);
            vmcntRelaxed++;
            llvm::errs() << "[postProcessISA] Removed redundant vmcnt(0) in "
                         << label << "\n";
          }
          break;
        }
      }
    }
  }

  // --- Pass 4b: Replace pre-GEMM1 FULL wait with lgkmcnt(0) ---
  {
    const std::string fullWait = "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)";
    size_t pos = 0;
    while (true) {
      size_t fw = loop.find(fullWait, pos);
      if (fw == std::string::npos) break;
      size_t fwEnd = loop.find('\n', fw);
      if (fwEnd == std::string::npos) { pos = fw + 1; continue; }
      fwEnd++;
      // Check if followed by v_mfma within 100 chars
      std::string afterChunk = loop.substr(fwEnd,
          std::min((size_t)100, loop.size() - fwEnd));
      if (afterChunk.find("v_mfma") != std::string::npos) {
        // Check if preceded by lgkmcnt(0) with no memory ops between
        size_t fwLineStart = loop.rfind('\n', fw);
        fwLineStart = (fwLineStart == std::string::npos) ? 0 : fwLineStart + 1;
        size_t lookbackStart = (fwLineStart > 400) ? fwLineStart - 400 : 0;
        std::string before = loop.substr(lookbackStart, fwLineStart - lookbackStart);
        size_t lgkm0 = before.rfind("s_waitcnt lgkmcnt(0)");
        if (lgkm0 != std::string::npos) {
          std::string gap = before.substr(lgkm0);
          if (gap.find("buffer_load") == std::string::npos &&
              gap.find("global_load") == std::string::npos &&
              gap.find("ds_read") == std::string::npos &&
              gap.find("ds_write") == std::string::npos &&
              gap.find("ds_permute") == std::string::npos) {
            loop.replace(fw, fullWait.size(), "s_waitcnt lgkmcnt(0)");
            vmcntRelaxed++;
            llvm::errs() << "[postProcessISA] Replaced pre-GEMM1 FULL wait"
                         << " with lgkmcnt(0) in " << label << "\n";
            pos = fw + 6;
            continue;
          }
        }
      }
      pos = fw + 1;
    }
  }

  // --- Pass 4c: Remove redundant consecutive vmcnt(0) ---
  unsigned vmcnt0Removed = 0;
  if (!isMultiBody) {
    const std::string vm0 = "s_waitcnt vmcnt(0)";
    size_t pos = 0;
    while (true) {
      size_t first = loop.find(vm0, pos);
      if (first == std::string::npos) break;
      std::string restFirst = loop.substr(first, 60);
      if (restFirst.find("expcnt") != std::string::npos) {
        pos = first + 20;
        continue;
      }
      size_t firstEnd = loop.find('\n', first);
      if (firstEnd == std::string::npos) break;
      firstEnd++;
      size_t second = loop.find(vm0, firstEnd);
      if (second == std::string::npos) break;
      std::string between = loop.substr(firstEnd, second - firstEnd);
      if (between.find("buffer_load") == std::string::npos &&
          between.find("global_load") == std::string::npos) {
        std::string restSecond = loop.substr(second, 60);
        if (restSecond.find("expcnt") == std::string::npos) {
          size_t secLineStart = loop.rfind('\n', second);
          secLineStart = (secLineStart == std::string::npos) ? 0 : secLineStart + 1;
          size_t secLineEnd = loop.find('\n', second);
          secLineEnd = (secLineEnd == std::string::npos) ? loop.size() : secLineEnd + 1;
          loop.erase(secLineStart, secLineEnd - secLineStart);
          vmcnt0Removed++;
          llvm::errs() << "[postProcessISA] Removed redundant vmcnt(0) in "
                       << label << "\n";
          continue;
        }
      }
      pos = firstEnd;
    }
  }

  // --- Pass 4d: Remove redundant lgkmcnt(0) using lgkm counter tracking ---
  // Tracks outstanding lgkm operations through the loop body. If a lgkmcnt(0)
  // is encountered when the counter is already 0, it's redundant and removed.
  // This handles the case where lgkmcnt(0) before s_barrier has no outstanding
  // lgkm ops (e.g., after O rescale with no ds/lds ops).
  unsigned lgkm0Removed = 0;
  {
    int lgkmOutstanding = 0;
    size_t pos = 0;
    while (pos < loop.size()) {
      size_t le = loop.find('\n', pos);
      if (le == std::string::npos) break;
      std::string cl = loop.substr(pos, le - pos);
      std::string clt = cl;
      size_t fns = clt.find_first_not_of(" \t");
      if (fns != std::string::npos) clt = clt.substr(fns);

      // Count lgkm-generating instructions
      if (clt.find("ds_read") == 0 || clt.find("ds_write") == 0 ||
          clt.find("ds_permute") == 0 || clt.find("s_load") == 0) {
        lgkmOutstanding++;
      } else if (clt.find("buffer_load") == 0 &&
                 cl.find(" lds") != std::string::npos) {
        lgkmOutstanding++;
      }

      // Check for lgkmcnt
      size_t lgkPos = cl.find("lgkmcnt(");
      if (lgkPos != std::string::npos) {
        size_t numStart = lgkPos + 8;
        size_t numEnd = cl.find(')', numStart);
        if (numEnd != std::string::npos) {
          int cnt = atoi(cl.substr(numStart, numEnd - numStart).c_str());
          if (cnt == 0 && lgkmOutstanding == 0) {
            // This lgkmcnt(0) is redundant — no outstanding lgkm ops
            // Check it's a standalone s_waitcnt lgkmcnt(0) (not a FULL wait)
            if (cl.find("expcnt") == std::string::npos &&
                cl.find("vmcnt") == std::string::npos) {
              loop.erase(pos, le + 1 - pos);
              lgkm0Removed++;
              llvm::errs() << "[postProcessISA] Removed redundant lgkmcnt(0) in "
                           << label << "\n";
              continue; // don't advance pos — content shifted
            }
          }
          if (cnt == 0) {
            lgkmOutstanding = 0;
          } else {
            lgkmOutstanding = std::min(lgkmOutstanding, cnt);
          }
        }
      }

      // s_barrier resets lgkm (all waves synchronized)
      if (clt.find("s_barrier") == 0) {
        lgkmOutstanding = 0;
      }

      pos = le + 1;
    }
  }

  // --- Pass 4e: Remove FULL wait redundant with preceding lgkmcnt(0)+vmcnt(0) ---
  // Pattern: lgkmcnt(0)...vmcnt(0)...FULL_WAIT with no LDS/VMEM between lgkmcnt(0) and FULL.
  // Also handles: lgkmcnt(0)...FULL_WAIT with no LDS/VMEM between (vmcnt already 0).
  unsigned fullWaitRemoved = 0;
  {
    const std::string fullWait = "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)";
    size_t pos = 0;
    while (true) {
      size_t fw = loop.find(fullWait, pos);
      if (fw == std::string::npos) break;
      size_t fwLineStart = loop.rfind('\n', fw);
      fwLineStart = (fwLineStart == std::string::npos) ? 0 : fwLineStart + 1;
      size_t fwLineEnd = loop.find('\n', fw);
      fwLineEnd = (fwLineEnd == std::string::npos) ? loop.size() : fwLineEnd + 1;

      // Look back up to 600 chars for a preceding lgkmcnt(0)
      size_t lookStart = (fwLineStart > 600) ? fwLineStart - 600 : 0;
      std::string before = loop.substr(lookStart, fwLineStart - lookStart);
      size_t lgkm0 = before.rfind("lgkmcnt(0)");
      if (lgkm0 != std::string::npos) {
        std::string gap = before.substr(lgkm0);
        bool hasLDS = gap.find("ds_read") != std::string::npos ||
                      gap.find("ds_write") != std::string::npos ||
                      gap.find("ds_permute") != std::string::npos;
        bool hasVMEM = gap.find("buffer_load") != std::string::npos ||
                       gap.find("global_load") != std::string::npos;
        if (!hasLDS && !hasVMEM) {
          loop.erase(fwLineStart, fwLineEnd - fwLineStart);
          fullWaitRemoved++;
          llvm::errs() << "[postProcessISA] Removed redundant FULL wait in "
                       << label << "\n";
          continue;
        }
      }
      pos = fw + 1;
    }
  }

  // --- Pass 9: Relax pre-GEMM1 lgkmcnt(0) to lgkmcnt(1) ---
  // DISABLED: ds_permute result (v73) is consumed immediately after the wait
  // (v_add_f32 v64, v72, v73), not later during GEMM1. Relaxing to lgkmcnt(1)
  // would read v73 before the permute completes → garbage results.
  unsigned lgkmRelaxed = 0;

  // --- Pass 5: Remove useless s_setprio 1 immediately followed by s_setprio 0 ---
  // Currently disabled: removing the pair causes slight regression.
  unsigned setprioRemoved = 0;
  if (false) {
    const std::string sp1 = "s_setprio 1";
    const std::string sp0 = "s_setprio 0";
    size_t pos = 0;
    while (true) {
      size_t p1 = loop.find(sp1, pos);
      if (p1 == std::string::npos) break;
      size_t p1End = loop.find('\n', p1);
      if (p1End == std::string::npos) { pos = p1 + 1; continue; }
      p1End++;
      // Skip whitespace/ASMEND lines to find next meaningful instruction
      std::string after = loop.substr(p1End, std::min((size_t)200, loop.size() - p1End));
      // Look for s_setprio 0 within 4 lines (may have ;;#ASMEND/ASMSTART between)
      size_t p0 = after.find(sp0);
      if (p0 != std::string::npos && p0 < 160) {
        // Check nothing meaningful between them (only ASM markers, whitespace)
        std::string between = after.substr(0, p0);
        bool onlyMarkers = true;
        for (const auto &ch : between) {
          if (ch != '\n' && ch != '\t' && ch != ' ' && ch != ';' && ch != '#' &&
              ch != 'A' && ch != 'S' && ch != 'M' && ch != 'E' && ch != 'N' &&
              ch != 'D' && ch != 'T' && ch != 'R') {
            onlyMarkers = false;
            break;
          }
        }
        if (onlyMarkers) {
          // Remove both s_setprio 1 line and s_setprio 0 line
          size_t p1LineStart = loop.rfind('\n', p1);
          if (p1LineStart == std::string::npos) p1LineStart = 0;
          else p1LineStart++;
          size_t p0Abs = p1End + p0;
          size_t p0End = loop.find('\n', p0Abs);
          if (p0End == std::string::npos) p0End = loop.size();
          else p0End++;
          // Remove from p1LineStart to p0End
          loop.erase(p1LineStart, p0End - p1LineStart);
          setprioRemoved++;
          llvm::errs() << "[postProcessISA] Removed useless s_setprio 1->0 pair in "
                       << label << "\n";
          pos = p1LineStart;
          continue;
        }
      }
      pos = p1 + 1;
    }
  }

  // --- Pass 6: Remove s_nop 0 between VALU instructions ---
  // GCNHazardRecognizer inserts s_nop 0 for MFMA->VALU read-after-write
  // hazards (11-instruction gap required). After scheduling, the gap may
  // already be satisfied, making the s_nop redundant. We remove s_nop 0
  // that are between two VALU (v_pk_add, v_pk_fma, v_add_f32, v_perm)
  // instructions where neither reads an MFMA result within 11 instructions.
  // Conservative: only remove s_nop 0 between v_pk_add_f32/v_add_f32 pairs
  // (the softmax reduction chain) where the gap from last MFMA is large.
  unsigned nopsRemoved = 0;
  {
    const std::string nop0 = "s_nop 0";
    size_t pos = 0;
    while (true) {
      size_t nop = loop.find(nop0, pos);
      if (nop == std::string::npos) break;
      size_t nopLineStart = loop.rfind('\n', nop);
      nopLineStart = (nopLineStart == std::string::npos) ? 0 : nopLineStart + 1;
      size_t nopLineEnd = loop.find('\n', nop);
      nopLineEnd = (nopLineEnd == std::string::npos) ? loop.size() : nopLineEnd + 1;

      // Check line before: must be VALU (v_pk_add, v_pk_fma, v_perm, v_add_f32)
      size_t prevLineEnd = nopLineStart;
      if (prevLineEnd > 0) prevLineEnd--;
      size_t prevLineStart = loop.rfind('\n', prevLineEnd);
      prevLineStart = (prevLineStart == std::string::npos) ? 0 : prevLineStart + 1;
      std::string prevLine = loop.substr(prevLineStart, prevLineEnd + 1 - prevLineStart);

      // Check line after: must be VALU
      std::string nextLine;
      if (nopLineEnd < loop.size()) {
        size_t nextEnd = loop.find('\n', nopLineEnd);
        if (nextEnd == std::string::npos) nextEnd = loop.size();
        nextLine = loop.substr(nopLineEnd, nextEnd - nopLineEnd);
      }

      bool prevIsValu = prevLine.find("v_pk_add_f32") != std::string::npos ||
                        prevLine.find("v_pk_fma_f32") != std::string::npos ||
                        prevLine.find("v_add_f32") != std::string::npos ||
                        prevLine.find("v_perm_b32") != std::string::npos ||
                        prevLine.find("v_mfma") != std::string::npos;
      bool nextIsValu = nextLine.find("v_pk_add_f32") != std::string::npos ||
                        nextLine.find("v_pk_fma_f32") != std::string::npos ||
                        nextLine.find("v_add_f32") != std::string::npos ||
                        nextLine.find("v_perm_b32") != std::string::npos ||
                        nextLine.find("v_mfma") != std::string::npos;

      // Check the immediately preceding non-NOP instruction is not v_mfma
      // (MFMA→VALU hazard needs 11-instruction gap, 1 NOP is insufficient)
      // Only skip removal if prev or next IS v_mfma (NOP between MFMA and VALU
      // is a different hazard handled by GCNHazardRecognizer).
      bool nearMfma = prevLine.find("v_mfma") != std::string::npos ||
                      nextLine.find("v_mfma") != std::string::npos;

      if (prevIsValu && nextIsValu && !nearMfma) {
        loop.erase(nopLineStart, nopLineEnd - nopLineStart);
        nopsRemoved++;
        continue;
      }
      pos = nopLineEnd;
    }
  }

  llvm::errs() << "[postProcessISA] " << label << ": moved " << cmpMoved
               << " v_cmp, yield=" << yieldInserted
               << ", nopFilled=" << nopFilled
               << ", vmcntRelaxed=" << vmcntRelaxed
               << ", vmcnt0Removed=" << vmcnt0Removed
               << ", lgkm0Removed=" << lgkm0Removed
               << ", lgkmRelaxed=" << lgkmRelaxed
               << ", fullWaitRemoved=" << fullWaitRemoved
               << ", setprioRemoved=" << setprioRemoved
               << ", nopsRemoved=" << nopsRemoved << "\n";

  return loop;
}

// Restructure GEMM2 MFMAs from column-cycling [1,1,...,1] to accumulator-grouped
// [4,4,4,4]. Each group of 4 MFMAs uses the same SrcC (accumulator) register,
// enabling back-to-back same-SrcC pipelining and better memory-compute overlap.
static std::string restructureGEMM2(std::string loop,
                                    const std::string &label) {
  auto extractVGPRRange = [](const std::string &s,
                             size_t startFrom) -> std::string {
    size_t pos = s.find("v[", startFrom);
    if (pos == std::string::npos)
      return "";
    size_t end = s.find(']', pos);
    if (end == std::string::npos)
      return "";
    return s.substr(pos, end - pos + 1);
  };

  size_t searchPos = 0;
  int bodiesProcessed = 0;

  while (true) {
    size_t sp0 = loop.find("s_setprio 0", searchPos);
    if (sp0 == std::string::npos)
      break;
    size_t sp0LineEnd = loop.find('\n', sp0);
    if (sp0LineEnd == std::string::npos)
      break;
    sp0LineEnd++;

    size_t endMarker = loop.find("v_pk_mul_f32", sp0LineEnd);
    size_t nextBarrier = loop.find("s_barrier", sp0LineEnd);
    if (endMarker == std::string::npos && nextBarrier == std::string::npos) {
      searchPos = sp0LineEnd;
      continue;
    }
    if (endMarker == std::string::npos ||
        (nextBarrier != std::string::npos && nextBarrier < endMarker))
      endMarker = nextBarrier;

    size_t endLineStart = loop.rfind('\n', endMarker);
    if (endLineStart == std::string::npos)
      endLineStart = 0;
    else
      endLineStart++;

    std::string region = loop.substr(sp0LineEnd, endLineStart - sp0LineEnd);

    // Split region into lines, tracking positions
    struct LineInfo {
      std::string text;
      size_t posInRegion;
    };
    std::vector<LineInfo> lines;
    {
      size_t pos = 0;
      while (pos < region.size()) {
        size_t eol = region.find('\n', pos);
        if (eol == std::string::npos) {
          if (pos < region.size())
            lines.push_back({region.substr(pos), pos});
          break;
        }
        lines.push_back({region.substr(pos, eol - pos + 1), pos});
        pos = eol + 1;
      }
    }

    // Collect MFMAs and ds_reads with their line indices
    struct MFMAEntry {
      int lineIdx;
      std::string dest, srcA, srcB;
    };
    struct DSReadEntry {
      int lineIdx;
      std::string dest;
      int offset;
    };

    std::vector<MFMAEntry> mfmas;
    std::vector<DSReadEntry> dsReads;

    for (int i = 0; i < (int)lines.size(); i++) {
      std::string stripped = lines[i].text;
      size_t cpos = stripped.find("//");
      if (cpos != std::string::npos)
        stripped = stripped.substr(0, cpos);

      if (stripped.find("v_mfma_f32_32x32x8_bf16") != std::string::npos) {
        MFMAEntry m;
        m.lineIdx = i;
        m.dest = extractVGPRRange(stripped, 0);
        size_t after1 = stripped.find(']') + 1;
        m.srcA = extractVGPRRange(stripped, after1);
        size_t after2 =
            stripped.find(']', stripped.find("v[", after1)) + 1;
        m.srcB = extractVGPRRange(stripped, after2);
        if (!m.dest.empty() && !m.srcA.empty() && !m.srcB.empty())
          mfmas.push_back(m);
      } else if (stripped.find("ds_read_b64") != std::string::npos) {
        DSReadEntry d;
        d.lineIdx = i;
        d.dest = extractVGPRRange(stripped, 0);
        size_t offPos = stripped.find("offset:");
        d.offset = (offPos != std::string::npos)
                       ? atoi(stripped.substr(offPos + 7).c_str())
                       : -1;
        dsReads.push_back(d);
      }
    }

    if (mfmas.size() != 16 || dsReads.size() < 16) {
      searchPos = endLineStart;
      continue;
    }

    // Build accOrder and verify 4 groups of 4
    std::vector<std::string> accOrder;
    std::map<std::string, int> accCount;
    for (auto &m : mfmas) {
      accCount[m.dest]++;
      if (std::find(accOrder.begin(), accOrder.end(), m.dest) ==
          accOrder.end())
        accOrder.push_back(m.dest);
    }
    if (accOrder.size() != 4) {
      searchPos = endLineStart;
      continue;
    }
    bool valid = true;
    for (auto &a : accOrder)
      if (accCount[a] != 4)
        valid = false;
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    // Determine SrcB registers from the 4 unique SrcB values used by acc0
    std::vector<std::string> srcBRegs;
    for (auto &m : mfmas) {
      if (m.dest == accOrder[0]) {
        srcBRegs.push_back(m.srcB);
      }
    }
    if (srcBRegs.size() != 4) {
      searchPos = endLineStart;
      continue;
    }

    // Verify all acc groups use the same SrcB order
    for (size_t ai = 1; ai < 4 && valid; ai++) {
      int ci = 0;
      for (auto &m : mfmas) {
        if (m.dest == accOrder[ai]) {
          if (m.srcB != srcBRegs[ci]) {
            valid = false;
            break;
          }
          ci++;
        }
      }
    }
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    // Determine original (accIdx, colIdx) for each MFMA position
    struct Assignment {
      int accIdx, colIdx;
    };
    std::vector<Assignment> origAssign(16);
    {
      std::map<std::string, int> accColCounter;
      for (int j = 0; j < 16; j++) {
        int ai = 0;
        for (int k = 0; k < 4; k++)
          if (mfmas[j].dest == accOrder[k])
            ai = k;
        int ci = accColCounter[mfmas[j].dest]++;
        origAssign[j] = {ai, ci};
      }
    }

    // Determine pBase and accStride from ds_reads
    // Build feed_map: which ds_read feeds which MFMA
    // Greedy: most recent ds_read to same register feeds the next MFMA using it
    std::vector<int> feedMap(16, -1); // feedMap[dsread_idx] = mfma_idx
    {
      // Interleave: walk through lines in order, track pending ds_reads
      std::map<std::string, int> pendingDsRead; // dest_reg -> dsread_idx
      int mfmaSeq = 0;
      int dsReadSeq = 0;
      for (int i = 0; i < (int)lines.size(); i++) {
        if (dsReadSeq < 16 && dsReads[dsReadSeq].lineIdx == i) {
          pendingDsRead[dsReads[dsReadSeq].dest] = dsReadSeq;
          dsReadSeq++;
        }
        if (mfmaSeq < 16 && mfmas[mfmaSeq].lineIdx == i) {
          auto it = pendingDsRead.find(mfmas[mfmaSeq].srcA);
          if (it != pendingDsRead.end()) {
            feedMap[it->second] = mfmaSeq;
            pendingDsRead.erase(it);
          }
          mfmaSeq++;
        }
      }
    }

    // Verify all feedMap entries are assigned
    for (int i = 0; i < 16; i++) {
      if (feedMap[i] < 0) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    // Build inverse feed map: invFeed[mfma_idx] = dsread_idx
    std::vector<int> invFeed(16, -1);
    for (int i = 0; i < 16; i++)
      invFeed[feedMap[i]] = i;

    // Compute pBase from the ds_read that feeds MFMA[0] (acc0, col0)
    // pBase[col] = offset for (acc0, col)
    int pBase[4];
    for (int c = 0; c < 4; c++) {
      // Find MFMA with origAssign = (0, c)
      for (int j = 0; j < 16; j++) {
        if (origAssign[j].accIdx == 0 && origAssign[j].colIdx == c) {
          pBase[c] = dsReads[invFeed[j]].offset;
          break;
        }
      }
    }

    // Compute accStride
    int accStride = 128;
    {
      // Find MFMA with origAssign = (1, 0) to get its ds_read offset
      for (int j = 0; j < 16; j++) {
        if (origAssign[j].accIdx == 1 && origAssign[j].colIdx == 0) {
          int off = dsReads[invFeed[j]].offset;
          int candidate = off - pBase[0];
          if (candidate > 0 && candidate < 1024)
            accStride = candidate;
          break;
        }
      }
    }

    // New assignment: [4,4,4,4] accumulator grouping
    // MFMA[j] -> new (group = j/4, col = j%4)
    std::vector<Assignment> newAssign(16);
    for (int j = 0; j < 16; j++)
      newAssign[j] = {j / 4, j % 4};

    // Log pre-modification state
    llvm::errs() << "[restructureGEMM2] Body " << bodiesProcessed << " in "
                 << label << "\n";
    llvm::errs() << "  accOrder:";
    for (auto &a : accOrder)
      llvm::errs() << " " << a;
    llvm::errs() << "\n  srcBRegs:";
    for (auto &s : srcBRegs)
      llvm::errs() << " " << s;
    llvm::errs() << "\n  pBase=[" << pBase[0] << "," << pBase[1] << ","
                 << pBase[2] << "," << pBase[3]
                 << "] accStride=" << accStride << "\n";
    llvm::errs() << "  feedMap:";
    for (int i = 0; i < 16; i++)
      llvm::errs() << " ds" << i << "->m" << feedMap[i];
    llvm::errs() << "\n  origAssign:";
    for (int j = 0; j < 16; j++)
      llvm::errs() << " m" << j << "=(a" << origAssign[j].accIdx
                   << ",c" << origAssign[j].colIdx << ")";
    llvm::errs() << "\n";

    // Verify offset formula for ALL 16 entries
    bool offsetFormulaOK = true;
    for (int j = 0; j < 16; j++) {
      int dsIdx = invFeed[j];
      int actualOff = dsReads[dsIdx].offset;
      int expectedOff = pBase[origAssign[j].colIdx] +
                        origAssign[j].accIdx * accStride;
      if (actualOff != expectedOff) {
        llvm::errs() << "  OFFSET MISMATCH: m" << j << " orig=(a"
                     << origAssign[j].accIdx << ",c" << origAssign[j].colIdx
                     << ") expected=" << expectedOff
                     << " actual=" << actualOff << "\n";
        offsetFormulaOK = false;
      }
    }
    if (!offsetFormulaOK) {
      llvm::errs() << "  SKIPPING body " << bodiesProcessed
                   << " due to offset formula mismatch\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    // Check for SrcB register overwrites between first and last MFMA.
    // Only writes BETWEEN MFMAs matter; writes after the last MFMA are safe.
    std::set<int> srcBVGPRs;
    for (auto &s : srcBRegs) {
      size_t lb = s.find('[');
      size_t colon = s.find(':');
      size_t rb = s.find(']');
      if (lb != std::string::npos && colon != std::string::npos &&
          rb != std::string::npos) {
        int lo = atoi(s.substr(lb + 1, colon - lb - 1).c_str());
        int hi = atoi(s.substr(colon + 1, rb - colon - 1).c_str());
        for (int r = lo; r <= hi; r++)
          srcBVGPRs.insert(r);
      }
    }

    std::set<int> trackedLines;
    for (auto &m : mfmas)
      trackedLines.insert(m.lineIdx);
    for (int d = 0; d < 16; d++)
      trackedLines.insert(dsReads[d].lineIdx);

    int lastMfmaLine = mfmas.back().lineIdx;

    auto checkVGPRConflict = [&](const std::string &stripped, int lineIdx) {
      // Try v[X:Y] range first
      std::string dest = extractVGPRRange(stripped, 0);
      if (!dest.empty()) {
        size_t lb = dest.find('[');
        size_t colon = dest.find(':');
        size_t rb = dest.find(']');
        if (lb != std::string::npos && colon != std::string::npos &&
            rb != std::string::npos) {
          int lo = atoi(dest.substr(lb + 1, colon - lb - 1).c_str());
          int hi = atoi(dest.substr(colon + 1, rb - colon - 1).c_str());
          for (int r = lo; r <= hi; r++) {
            if (srcBVGPRs.count(r)) {
              llvm::errs() << "  SrcB CONFLICT: v" << r
                           << " at line " << lineIdx << ": " << stripped << "\n";
              return true;
            }
          }
        }
        return false;
      }
      // Try single vN
      size_t vpos = stripped.find(" v");
      if (vpos != std::string::npos) {
        vpos++;
        size_t numStart = vpos + 1;
        size_t numEnd = numStart;
        while (numEnd < stripped.size() && isdigit(stripped[numEnd]))
          numEnd++;
        if (numEnd > numStart) {
          int vgpr = atoi(stripped.substr(numStart, numEnd - numStart).c_str());
          if (srcBVGPRs.count(vgpr)) {
            llvm::errs() << "  SrcB CONFLICT: v" << vgpr
                         << " at line " << lineIdx << ": " << stripped << "\n";
            return true;
          }
        }
      }
      return false;
    };

    bool srcBOverwritten = false;
    for (int i = 0; i <= lastMfmaLine && !srcBOverwritten; i++) {
      if (trackedLines.count(i))
        continue;
      std::string stripped = lines[i].text;
      size_t cpos = stripped.find("//");
      if (cpos != std::string::npos)
        stripped = stripped.substr(0, cpos);

      // Detect ANY instruction that writes to a VGPR
      bool isVGPRWrite = false;
      // Loads (buffer_load without lds, global_load, flat_load)
      if (stripped.find("buffer_load") != std::string::npos &&
          stripped.find(" lds") == std::string::npos)
        isVGPRWrite = true;
      if (stripped.find("global_load") != std::string::npos ||
          stripped.find("flat_load") != std::string::npos)
        isVGPRWrite = true;
      // VALU writes (v_* except v_mfma which is tracked)
      if (!isVGPRWrite && stripped.find("v_") != std::string::npos)
        isVGPRWrite = true;
      // ds_permute writes to first VGPR operand
      if (stripped.find("ds_permute") != std::string::npos)
        isVGPRWrite = true;

      if (!isVGPRWrite)
        continue;

      srcBOverwritten = checkVGPRConflict(stripped, i);
    }

    if (srcBOverwritten) {
      llvm::errs() << "  SKIPPING body " << bodiesProcessed
                   << " due to SrcB register overwrite\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    // Apply in-place modifications: change MFMA operands and ds_read offsets
    int modified = 0;
    for (int j = 0; j < 16; j++) {
      int newAcc = newAssign[j].accIdx;
      int newCol = newAssign[j].colIdx;
      std::string newAccReg = accOrder[newAcc];
      std::string newSrcB = srcBRegs[newCol];
      int newOffset = pBase[newCol] + newAcc * accStride;

      int mfmaLine = mfmas[j].lineIdx;
      std::string &mline = lines[mfmaLine].text;
      std::string oldDest = mfmas[j].dest;
      std::string oldSrcB = mfmas[j].srcB;

      llvm::errs() << "  m" << j << ": (a" << origAssign[j].accIdx << ",c"
                   << origAssign[j].colIdx << ")->(a" << newAcc << ",c"
                   << newCol << ")";

      if (oldDest != newAccReg || oldSrcB != newSrcB) {
        std::string newInst = "\tv_mfma_f32_32x32x8_bf16 " + newAccReg +
                              ", " + mfmas[j].srcA + ", " + newSrcB + ", " +
                              newAccReg + "\n";
        llvm::errs() << " MFMA: " << oldDest << "," << oldSrcB << " -> "
                     << newAccReg << "," << newSrcB;
        mline = newInst;
        modified++;
      }

      int dsIdx = invFeed[j];
      if (dsIdx >= 0) {
        int dsLine = dsReads[dsIdx].lineIdx;
        std::string &dline = lines[dsLine].text;
        int oldOffset = dsReads[dsIdx].offset;
        if (oldOffset != newOffset) {
          std::string oldOffStr = "offset:" + std::to_string(oldOffset);
          std::string newOffStr = "offset:" + std::to_string(newOffset);
          size_t offPos = dline.find(oldOffStr);
          if (offPos != std::string::npos) {
            dline.replace(offPos, oldOffStr.size(), newOffStr);
            llvm::errs() << " DS: " << oldOffset << "->" << newOffset;
            modified++;
          } else {
            llvm::errs() << " DS: OFFSET NOT FOUND in line!";
          }
        }
      }
      llvm::errs() << "\n";
    }

    if (modified == 0) {
      llvm::errs() << "  SKIPPING body " << bodiesProcessed
                   << " (already grouped)\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    llvm::errs() << "  Total modified: " << modified << "\n";

    // Reconstruct the region from modified lines
    std::string newRegion;
    for (auto &li : lines)
      newRegion += li.text;

    loop.replace(sp0LineEnd, endLineStart - sp0LineEnd, newRegion);
    searchPos = sp0LineEnd + newRegion.size();
    bodiesProcessed++;
  }

  if (bodiesProcessed > 0)
    llvm::errs() << "[postProcessISA] restructureGEMM2: processed "
                 << bodiesProcessed << " bodies in " << label << "\n";

  return loop;
}

// --- Pass: restructureGEMM2_V4 ---
// Reorder instructions: group ds_reads into 4-read batches, each batch
// followed by 4 MFMAs that consume them (same SrcC, back-to-back).
// Filler is kept in original order between groups 0 and 1.
// SrcC transition gaps for groups 1-3 are padded with s_nop.
static std::string restructureGEMM2_V4(std::string loop,
                                       const std::string &label) {
  auto extractVGPRRange = [](const std::string &s,
                             size_t startFrom) -> std::string {
    size_t pos = s.find("v[", startFrom);
    if (pos == std::string::npos)
      return "";
    size_t end = s.find(']', pos);
    if (end == std::string::npos)
      return "";
    return s.substr(pos, end - pos + 1);
  };

  size_t searchPos = 0;
  int bodiesProcessed = 0;

  while (true) {
    size_t sp0 = loop.find("s_setprio 0", searchPos);
    if (sp0 == std::string::npos)
      break;
    size_t sp0LineEnd = loop.find('\n', sp0);
    if (sp0LineEnd == std::string::npos)
      break;
    sp0LineEnd++;

    size_t endMarker = loop.find("v_pk_mul_f32", sp0LineEnd);
    size_t nextBarrier = loop.find("s_barrier", sp0LineEnd);
    if (endMarker == std::string::npos && nextBarrier == std::string::npos) {
      searchPos = sp0LineEnd;
      continue;
    }
    if (endMarker == std::string::npos ||
        (nextBarrier != std::string::npos && nextBarrier < endMarker))
      endMarker = nextBarrier;

    size_t endLineStart = loop.rfind('\n', endMarker);
    if (endLineStart == std::string::npos)
      endLineStart = 0;
    else
      endLineStart++;

    std::string region = loop.substr(sp0LineEnd, endLineStart - sp0LineEnd);

    struct LineInfo {
      std::string text;
    };
    std::vector<LineInfo> lines;
    {
      size_t pos = 0;
      while (pos < region.size()) {
        size_t eol = region.find('\n', pos);
        if (eol == std::string::npos) {
          if (pos < region.size())
            lines.push_back({region.substr(pos)});
          break;
        }
        lines.push_back({region.substr(pos, eol - pos + 1)});
        pos = eol + 1;
      }
    }

    struct MFMAEntry {
      int lineIdx;
      std::string dest, srcA, srcB;
    };
    struct DSReadEntry {
      int lineIdx;
      std::string dest;
      int offset;
    };

    std::vector<MFMAEntry> mfmas;
    std::vector<DSReadEntry> dsReads;

    for (int i = 0; i < (int)lines.size(); i++) {
      std::string stripped = lines[i].text;
      size_t cpos = stripped.find("//");
      if (cpos != std::string::npos)
        stripped = stripped.substr(0, cpos);

      if (stripped.find("v_mfma_f32_32x32x8_bf16") != std::string::npos) {
        MFMAEntry m;
        m.lineIdx = i;
        m.dest = extractVGPRRange(stripped, 0);
        size_t after1 = stripped.find(']') + 1;
        m.srcA = extractVGPRRange(stripped, after1);
        size_t after2 =
            stripped.find(']', stripped.find("v[", after1)) + 1;
        m.srcB = extractVGPRRange(stripped, after2);
        if (!m.dest.empty() && !m.srcA.empty() && !m.srcB.empty())
          mfmas.push_back(m);
      } else if (stripped.find("ds_read_b64") != std::string::npos) {
        DSReadEntry d;
        d.lineIdx = i;
        d.dest = extractVGPRRange(stripped, 0);
        size_t offPos = stripped.find("offset:");
        d.offset = (offPos != std::string::npos)
                       ? atoi(stripped.substr(offPos + 7).c_str())
                       : -1;
        dsReads.push_back(d);
      }
    }

    if (mfmas.size() != 16 || dsReads.size() < 16) {
      searchPos = endLineStart;
      continue;
    }

    // Build accOrder and verify 4 groups of 4
    std::vector<std::string> accOrder;
    std::map<std::string, int> accCount;
    for (auto &m : mfmas) {
      accCount[m.dest]++;
      if (std::find(accOrder.begin(), accOrder.end(), m.dest) ==
          accOrder.end())
        accOrder.push_back(m.dest);
    }
    if (accOrder.size() != 4) {
      searchPos = endLineStart;
      continue;
    }
    bool valid = true;
    for (auto &a : accOrder)
      if (accCount[a] != 4)
        valid = false;
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    // Determine SrcB registers per column (from acc0's 4 MFMAs)
    std::vector<std::string> srcBRegs;
    for (auto &m : mfmas) {
      if (m.dest == accOrder[0])
        srcBRegs.push_back(m.srcB);
    }
    if (srcBRegs.size() != 4) {
      searchPos = endLineStart;
      continue;
    }

    // Verify all acc groups use the same SrcB order
    for (size_t ai = 1; ai < 4 && valid; ai++) {
      int ci = 0;
      for (auto &m : mfmas) {
        if (m.dest == accOrder[ai]) {
          if (m.srcB != srcBRegs[ci]) {
            valid = false;
            break;
          }
          ci++;
        }
      }
    }
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    // Determine original (accIdx, colIdx) for each MFMA
    struct Assignment {
      int accIdx, colIdx;
    };
    std::vector<Assignment> origAssign(16);
    {
      std::map<std::string, int> accColCounter;
      for (int j = 0; j < 16; j++) {
        int ai = 0;
        for (int k = 0; k < 4; k++)
          if (mfmas[j].dest == accOrder[k])
            ai = k;
        int ci = accColCounter[mfmas[j].dest]++;
        origAssign[j] = {ai, ci};
      }
    }

    // Build feed_map and inverse feed_map (same as V1)
    std::vector<int> feedMap(16, -1);
    {
      std::map<std::string, int> pendingDsRead;
      int mfmaSeq = 0;
      int dsReadSeq = 0;
      for (int i = 0; i < (int)lines.size(); i++) {
        if (dsReadSeq < 16 && dsReads[dsReadSeq].lineIdx == i) {
          pendingDsRead[dsReads[dsReadSeq].dest] = dsReadSeq;
          dsReadSeq++;
        }
        if (mfmaSeq < 16 && mfmas[mfmaSeq].lineIdx == i) {
          auto it = pendingDsRead.find(mfmas[mfmaSeq].srcA);
          if (it != pendingDsRead.end()) {
            feedMap[it->second] = mfmaSeq;
            pendingDsRead.erase(it);
          }
          mfmaSeq++;
        }
      }
    }

    for (int i = 0; i < 16; i++) {
      if (feedMap[i] < 0) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      searchPos = endLineStart;
      continue;
    }

    std::vector<int> invFeed(16, -1);
    for (int i = 0; i < 16; i++)
      invFeed[feedMap[i]] = i;

    // Compute pBase and accStride
    int pBase[4];
    for (int c = 0; c < 4; c++) {
      for (int j = 0; j < 16; j++) {
        if (origAssign[j].accIdx == 0 && origAssign[j].colIdx == c) {
          pBase[c] = dsReads[invFeed[j]].offset;
          break;
        }
      }
    }

    int accStride = 128;
    {
      for (int j = 0; j < 16; j++) {
        if (origAssign[j].accIdx == 1 && origAssign[j].colIdx == 0) {
          int off = dsReads[invFeed[j]].offset;
          int candidate = off - pBase[0];
          if (candidate > 0 && candidate < 1024)
            accStride = candidate;
          break;
        }
      }
    }

    // Verify offset formula
    bool offsetFormulaOK = true;
    for (int j = 0; j < 16; j++) {
      int dsIdx = invFeed[j];
      int actualOff = dsReads[dsIdx].offset;
      int expectedOff = pBase[origAssign[j].colIdx] +
                        origAssign[j].accIdx * accStride;
      if (actualOff != expectedOff) {
        offsetFormulaOK = false;
        break;
      }
    }
    if (!offsetFormulaOK) {
      llvm::errs() << "[restructureGEMM2_V4] SKIP body " << bodiesProcessed
                   << " in " << label << ": offset formula mismatch\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    // Extract address register from first ds_read (e.g., "v126")
    std::string addrReg;
    {
      std::string &firstRead = lines[dsReads[0].lineIdx].text;
      size_t commaPos = firstRead.find(',');
      if (commaPos != std::string::npos) {
        size_t addrStart = firstRead.find_first_not_of(" \t", commaPos + 1);
        size_t addrEnd = firstRead.find_first_of(" \t", addrStart);
        if (addrStart != std::string::npos && addrEnd != std::string::npos)
          addrReg = firstRead.substr(addrStart, addrEnd - addrStart);
      }
    }
    if (addrReg.empty()) {
      searchPos = endLineStart;
      continue;
    }

    // Check SrcB register overwrites between first and last MFMA
    std::set<int> srcBVGPRs;
    for (auto &s : srcBRegs) {
      size_t lb = s.find('[');
      size_t colon = s.find(':');
      size_t rb = s.find(']');
      if (lb != std::string::npos && colon != std::string::npos &&
          rb != std::string::npos) {
        int lo = atoi(s.substr(lb + 1, colon - lb - 1).c_str());
        int hi = atoi(s.substr(colon + 1, rb - colon - 1).c_str());
        for (int r = lo; r <= hi; r++)
          srcBVGPRs.insert(r);
      }
    }

    // For V4: SrcB must not be overwritten by ANY filler instruction
    // (since all filler runs before groups 1-3 which use SrcB)
    std::set<int> trackedLines;
    for (auto &m : mfmas)
      trackedLines.insert(m.lineIdx);
    for (int d = 0; d < 16; d++)
      trackedLines.insert(dsReads[d].lineIdx);

    bool srcBOverwritten = false;
    int lastMfmaLine = mfmas.back().lineIdx;
    for (int i = 0; i <= lastMfmaLine && !srcBOverwritten; i++) {
      if (trackedLines.count(i))
        continue;
      std::string stripped = lines[i].text;
      size_t cpos = stripped.find("//");
      if (cpos != std::string::npos)
        stripped = stripped.substr(0, cpos);

      bool isVGPRWrite = false;
      if (stripped.find("buffer_load") != std::string::npos &&
          stripped.find(" lds") == std::string::npos)
        isVGPRWrite = true;
      if (stripped.find("global_load") != std::string::npos ||
          stripped.find("flat_load") != std::string::npos)
        isVGPRWrite = true;
      if (!isVGPRWrite && stripped.find("v_") != std::string::npos)
        isVGPRWrite = true;
      if (stripped.find("ds_permute") != std::string::npos)
        isVGPRWrite = true;
      if (!isVGPRWrite)
        continue;

      // Check range dest
      std::string dest = extractVGPRRange(stripped, 0);
      if (!dest.empty()) {
        size_t lb = dest.find('[');
        size_t colon = dest.find(':');
        size_t rb = dest.find(']');
        if (lb != std::string::npos && colon != std::string::npos &&
            rb != std::string::npos) {
          int lo = atoi(dest.substr(lb + 1, colon - lb - 1).c_str());
          int hi = atoi(dest.substr(colon + 1, rb - colon - 1).c_str());
          for (int r = lo; r <= hi; r++) {
            if (srcBVGPRs.count(r)) {
              llvm::errs() << "[restructureGEMM2_V4] SrcB CONFLICT v" << r
                           << " at line " << i << "\n";
              srcBOverwritten = true;
            }
          }
        }
      } else {
        size_t vpos = stripped.find(" v");
        if (vpos != std::string::npos) {
          vpos++;
          size_t numStart = vpos + 1;
          size_t numEnd = numStart;
          while (numEnd < stripped.size() && isdigit(stripped[numEnd]))
            numEnd++;
          if (numEnd > numStart) {
            int vgpr =
                atoi(stripped.substr(numStart, numEnd - numStart).c_str());
            if (srcBVGPRs.count(vgpr)) {
              llvm::errs() << "[restructureGEMM2_V4] SrcB CONFLICT v" << vgpr
                           << " at line " << i << "\n";
              srcBOverwritten = true;
            }
          }
        }
      }
    }

    if (srcBOverwritten) {
      llvm::errs() << "[restructureGEMM2_V4] SKIP body " << bodiesProcessed
                   << " in " << label << " due to SrcB overwrite\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    // Scan filler for VGPR writes to build "unsafe" register set
    std::set<int> unsafeVGPRs;
    for (int i = 0; i < (int)lines.size(); i++) {
      if (trackedLines.count(i))
        continue;
      std::string stripped = lines[i].text;
      size_t cpos = stripped.find("//");
      if (cpos != std::string::npos)
        stripped = stripped.substr(0, cpos);
      if (stripped.find("v_") == std::string::npos &&
          stripped.find("ds_permute") == std::string::npos &&
          stripped.find("buffer_load") == std::string::npos)
        continue;
      // Extract dest VGPR
      std::string dest = extractVGPRRange(stripped, 0);
      if (!dest.empty()) {
        size_t lb = dest.find('[');
        size_t colon = dest.find(':');
        size_t rb = dest.find(']');
        if (lb != std::string::npos && colon != std::string::npos &&
            rb != std::string::npos) {
          int lo = atoi(dest.substr(lb + 1, colon - lb - 1).c_str());
          int hi = atoi(dest.substr(colon + 1, rb - colon - 1).c_str());
          for (int r = lo; r <= hi; r++)
            unsafeVGPRs.insert(r);
        }
      } else {
        size_t vpos = stripped.find(" v");
        if (vpos != std::string::npos) {
          vpos++;
          size_t numStart = vpos + 1;
          size_t numEnd = numStart;
          while (numEnd < stripped.size() && isdigit(stripped[numEnd]))
            numEnd++;
          if (numEnd > numStart) {
            int vgpr =
                atoi(stripped.substr(numStart, numEnd - numStart).c_str());
            unsafeVGPRs.insert(vgpr);
          }
        }
      }
    }

    // Choose 4 even-aligned register pairs from existing read dests that
    // are NOT in the unsafe set
    std::set<std::string> allReadDests;
    for (int d = 0; d < 16; d++)
      allReadDests.insert(dsReads[d].dest);

    std::vector<std::string> readPairs;
    for (auto &rd : allReadDests) {
      size_t lb = rd.find('[');
      size_t colon = rd.find(':');
      if (lb == std::string::npos || colon == std::string::npos)
        continue;
      int lo = atoi(rd.substr(lb + 1, colon - lb - 1).c_str());
      int hi = lo + 1;
      if (lo % 2 != 0)
        continue;
      if (unsafeVGPRs.count(lo) || unsafeVGPRs.count(hi))
        continue;
      readPairs.push_back(rd);
      if (readPairs.size() == 4)
        break;
    }

    if (readPairs.size() < 4) {
      llvm::errs() << "[restructureGEMM2_V4] SKIP body " << bodiesProcessed
                   << " in " << label << ": only " << readPairs.size()
                   << " safe register pairs found\n";
      bodiesProcessed++;
      searchPos = endLineStart;
      continue;
    }

    // Separate filler into mainFiller (before/between MFMAs) and
    // postFiller (after last MFMA: v_exp, s_nop)
    std::vector<std::string> mainFiller, postFiller;
    bool pastLastMfma = false;
    for (int i = 0; i < (int)lines.size(); i++) {
      if (i > lastMfmaLine)
        pastLastMfma = true;
      if (trackedLines.count(i))
        continue;
      if (pastLastMfma) {
        postFiller.push_back(lines[i].text);
      } else {
        mainFiller.push_back(lines[i].text);
      }
    }

    llvm::errs() << "[restructureGEMM2_V4] Body " << bodiesProcessed << " in "
                 << label << "\n";
    llvm::errs() << "  accOrder:";
    for (auto &a : accOrder)
      llvm::errs() << " " << a;
    llvm::errs() << "\n  srcBRegs:";
    for (auto &s : srcBRegs)
      llvm::errs() << " " << s;
    llvm::errs() << "\n  readPairs:";
    for (auto &rp : readPairs)
      llvm::errs() << " " << rp;
    llvm::errs() << "\n  pBase=[" << pBase[0] << "," << pBase[1] << ","
                 << pBase[2] << "," << pBase[3]
                 << "] accStride=" << accStride << "\n";
    llvm::errs() << "  mainFiller: " << mainFiller.size()
                 << " lines, postFiller: " << postFiller.size() << " lines\n";

    // Generate restructured region (V5 layout):
    // Group 0: reads → lgkmcnt(3) → MFMAs → filler
    // Groups 1-3: reads → 5 s_nop (SrcC padding) → lgkmcnt(3) → MFMAs
    // lgkmcnt(3) waits for oldest read only; MFMA exec hides remaining reads.
    std::string newRegion;

    for (int g = 0; g < 4; g++) {
      // Emit 4 ds_reads for this accumulator group
      for (int c = 0; c < 4; c++) {
        int offset = pBase[c] + g * accStride;
        newRegion += "\tds_read_b64 " + readPairs[c] + ", " + addrReg +
                     " offset:" + std::to_string(offset) + "\n";
      }

      if (g > 0) {
        // Groups 1-3: SrcC transition padding before MFMAs.
        // 4 reads (just emitted) + 5 s_nop = 9 instructions for SrcC gap.
        for (int n = 0; n < 5; n++)
          newRegion += "\ts_nop 0\n";
      }

      // lgkmcnt(3): wait for oldest read only. The remaining 3 reads
      // complete during the first MFMA's 32-cycle execution window.
      newRegion += "\ts_waitcnt lgkmcnt(3)\n";

      // Emit 4 MFMAs (back-to-back, same SrcC)
      for (int c = 0; c < 4; c++) {
        newRegion += "\tv_mfma_f32_32x32x8_bf16 " + accOrder[g] + ", " +
                     readPairs[c] + ", " + srcBRegs[c] + ", " + accOrder[g] +
                     "\n";
      }

      // After group 0: emit ALL main filler in original order
      if (g == 0) {
        for (auto &fl : mainFiller)
          newRegion += fl;
      }
    }

    // Emit post-filler (v_exp, s_nop 6, etc.)
    for (auto &pf : postFiller)
      newRegion += pf;

    // Replace region
    loop.replace(sp0LineEnd, endLineStart - sp0LineEnd, newRegion);
    searchPos = sp0LineEnd + newRegion.size();
    bodiesProcessed++;
    llvm::errs() << "  APPLIED V4 restructuring\n";
  }

  if (bodiesProcessed > 0)
    llvm::errs() << "[postProcessISA] restructureGEMM2_V4: processed "
                 << bodiesProcessed << " bodies in " << label << "\n";

  return loop;
}

// --- Pass: restructureGEMM2_V3 ---
// Remap ds_read destinations to unique even-aligned registers, then reorder
// ds_reads + MFMAs into accumulator-grouped blocks with filler interleaved.
// Group N reads are issued before Group N-1 MFMAs (latency hidden by filler).
// Eliminates ~6 lgkmcnt stalls per body + reduces SrcC transition penalties.
static std::string restructureGEMM2_V3(std::string loop, const std::string &label,
                                       int &maxNewVgprOut) {
  int bodiesProcessed = 0;
  int maxNewVgpr = 0;

  size_t searchPos = 0;
  while (true) {
    size_t sp0 = loop.find("s_setprio 0\n", searchPos);
    if (sp0 == std::string::npos) break;

    size_t mfmaBefore = loop.rfind("v_mfma_f32_32x32x8_bf16", sp0);
    if (mfmaBefore == std::string::npos || sp0 - mfmaBefore > 300) {
      searchPos = sp0 + 12;
      continue;
    }

    size_t asmEnd = loop.find(";;#ASMEND", sp0);
    if (asmEnd == std::string::npos) { searchPos = sp0 + 12; continue; }
    size_t regionStart = loop.find('\n', asmEnd);
    if (regionStart == std::string::npos) break;
    regionStart++;

    // Region ends at v_exp_f32 (correction factor computation, must stay after MFMAs)
    size_t vexpPos = std::string::npos;
    {
      size_t p = regionStart;
      // Find v_exp_f32 that comes after MFMA instructions (the O correction exp)
      size_t lastMfma = regionStart;
      while (true) {
        size_t m = loop.find("v_mfma_f32_32x32x8_bf16", lastMfma + 1);
        if (m == std::string::npos || m > regionStart + 8000) break;
        // Check this MFMA is before the next s_setprio or s_barrier
        size_t nextSP = loop.find("s_setprio", m);
        size_t nextBar = loop.find("s_barrier", m);
        size_t boundary = std::min(nextSP, nextBar);
        if (boundary != std::string::npos && boundary < m) break;
        lastMfma = m;
      }
      // Find v_exp_f32 after the last MFMA
      size_t ve = loop.find("v_exp_f32", lastMfma);
      if (ve != std::string::npos) {
        size_t veLineStart = loop.rfind('\n', ve);
        vexpPos = (veLineStart == std::string::npos) ? ve : veLineStart + 1;
      }
    }
    if (vexpPos == std::string::npos || vexpPos <= regionStart) {
      searchPos = sp0 + 12;
      continue;
    }
    size_t regionEnd = vexpPos;

    std::string region = loop.substr(regionStart, regionEnd - regionStart);

    // Parse all lines
    struct Line {
      std::string text;
      int type; // 0=ds_read, 1=mfma, 2=lgkmcnt, 3=other
      int destReg, offset;   // ds_read
      int accReg, srcAReg, srcBReg; // mfma
    };
    std::vector<Line> allLines;
    {
      size_t pos = 0;
      while (pos < region.size()) {
        size_t eol = region.find('\n', pos);
        if (eol == std::string::npos) eol = region.size();
        std::string lt = region.substr(pos, eol - pos);
        Line L;
        L.text = lt; L.type = 3;
        L.destReg = -1; L.offset = 0;
        L.accReg = -1; L.srcAReg = -1; L.srcBReg = -1;

        if (lt.find("ds_read_b64 v[") != std::string::npos) {
          L.type = 0;
          size_t vs = lt.find("v[") + 2;
          L.destReg = atoi(lt.c_str() + vs);
          size_t os = lt.find("offset:");
          if (os != std::string::npos) L.offset = atoi(lt.c_str() + os + 7);
        } else if (lt.find("v_mfma_f32_32x32x8_bf16") != std::string::npos) {
          L.type = 1;
          std::vector<int> vr;
          size_t vp = 0;
          while (true) {
            size_t v = lt.find("v[", vp);
            if (v == std::string::npos) break;
            vr.push_back(atoi(lt.c_str() + v + 2));
            vp = v + 2;
          }
          if (vr.size() >= 4) {
            L.accReg = vr[0]; L.srcAReg = vr[1]; L.srcBReg = vr[2];
          }
        } else if (lt.find("s_waitcnt lgkmcnt(") != std::string::npos) {
          L.type = 2;
        }
        allLines.push_back(L);
        pos = eol + 1;
      }
    }

    // Collect ds_reads and MFMAs
    struct DSRead { int idx; std::string text; int destReg; int offset; int accGroup; };
    struct MFMA { int idx; std::string text; int accReg; int srcAReg; int srcBReg; int accGroup; };
    std::vector<DSRead> reads;
    std::vector<MFMA> mfmas;
    for (int i = 0; i < (int)allLines.size(); i++) {
      if (allLines[i].type == 0)
        reads.push_back({i, allLines[i].text, allLines[i].destReg, allLines[i].offset, -1});
      else if (allLines[i].type == 1)
        mfmas.push_back({i, allLines[i].text, allLines[i].accReg, allLines[i].srcAReg, allLines[i].srcBReg, -1});
    }

    if ((int)reads.size() < 16 || (int)mfmas.size() < 16) {
      searchPos = regionEnd; continue;
    }

    // Simulate register tracking to pair ds_reads with MFMAs
    std::map<int, int> liveReg; // reg -> ds_read index in `reads`
    int readIdx = 0, mfmaIdx = 0;
    std::map<int, int> mfmaToRead; // mfma index -> read index
    for (int i = 0; i < (int)allLines.size(); i++) {
      if (allLines[i].type == 0 && readIdx < (int)reads.size()) {
        liveReg[reads[readIdx].destReg] = readIdx;
        readIdx++;
      } else if (allLines[i].type == 1 && mfmaIdx < (int)mfmas.size()) {
        auto it = liveReg.find(mfmas[mfmaIdx].srcAReg);
        if (it != liveReg.end())
          mfmaToRead[mfmaIdx] = it->second;
        mfmaIdx++;
      }
    }

    // Determine accumulator groups: unique accReg values in MFMA order
    std::vector<int> accOrder;
    std::map<int, int> accToGroup;
    for (auto &m : mfmas) {
      if (accToGroup.find(m.accReg) == accToGroup.end()) {
        int g = (int)accOrder.size();
        accToGroup[m.accReg] = g;
        accOrder.push_back(m.accReg);
      }
      m.accGroup = accToGroup[m.accReg];
    }
    if ((int)accOrder.size() != 4) { searchPos = regionEnd; continue; }

    // Assign accGroup to reads via mfmaToRead mapping
    for (int mi = 0; mi < (int)mfmas.size(); mi++) {
      auto it = mfmaToRead.find(mi);
      if (it != mfmaToRead.end())
        reads[it->second].accGroup = mfmas[mi].accGroup;
    }

    // Group reads and MFMAs by accGroup
    std::vector<DSRead> groupReads[4];
    std::vector<MFMA> groupMfmas[4];
    for (auto &r : reads) if (r.accGroup >= 0 && r.accGroup < 4) groupReads[r.accGroup].push_back(r);
    for (auto &m : mfmas) if (m.accGroup >= 0 && m.accGroup < 4) groupMfmas[m.accGroup].push_back(m);

    // Verify: each group should have 4 reads and 4 MFMAs
    bool valid = true;
    for (int g = 0; g < 4; g++)
      if ((int)groupReads[g].size() != 4 || (int)groupMfmas[g].size() != 4) valid = false;
    if (!valid) { searchPos = regionEnd; continue; }

    // Assign unique even-aligned registers for groups 1-3 (group 0 keeps originals)
    // Build pool of even-aligned regs not used in this body
    std::set<int> usedRegs;
    for (auto &L : allLines) {
      size_t vp = 0;
      while (true) {
        size_t v = L.text.find("v[", vp);
        if (v == std::string::npos) break;
        int r = atoi(L.text.c_str() + v + 2);
        usedRegs.insert(r); usedRegs.insert(r+1);
        if (L.text.size() > v+2) {
          size_t colon = L.text.find(':', v+2);
          if (colon != std::string::npos) {
            int r2 = atoi(L.text.c_str() + colon + 1);
            for (int x = r; x <= r2; x++) usedRegs.insert(x);
          }
        }
        vp = v + 2;
      }
      // Also check bare vN references
      for (int r = 0; r < 256; r++) {
        std::string pat = "v" + std::to_string(r);
        if (L.text.find(pat + ",") != std::string::npos ||
            L.text.find(pat + " ") != std::string::npos)
          usedRegs.insert(r);
      }
    }

    // Find free even-aligned pairs (checking full kernel scope would be better but
    // this is conservative enough for the GEMM2 region)
    std::vector<int> freePool;
    for (int r = 220; r <= 252; r += 2) {
      if (usedRegs.find(r) == usedRegs.end() && usedRegs.find(r+1) == usedRegs.end())
        freePool.push_back(r);
    }

    // Need up to 12 free pairs (4 per group × 3 groups to remap)
    // But some reads in groups 1-3 might already have unique registers
    int poolUsed = 0;
    for (int g = 1; g < 4; g++) {
      for (auto &r : groupReads[g]) {
        // Check if this register is used by another read in any group
        bool needsRemap = false;
        for (int g2 = 0; g2 < 4; g2++) {
          if (g2 == g) continue;
          for (auto &r2 : groupReads[g2]) {
            if (r2.destReg == r.destReg) { needsRemap = true; break; }
          }
          if (needsRemap) break;
        }
        // Also check if used by another read in same group
        if (!needsRemap) {
          for (auto &r2 : groupReads[g]) {
            if (&r2 != &r && r2.destReg == r.destReg) { needsRemap = true; break; }
          }
        }

        if (needsRemap && poolUsed < (int)freePool.size()) {
          int newReg = freePool[poolUsed++];
          if (newReg + 2 > maxNewVgpr) maxNewVgpr = newReg + 2;

          std::string oldS = "v[" + std::to_string(r.destReg) + ":" + std::to_string(r.destReg+1) + "]";
          std::string newS = "v[" + std::to_string(newReg) + ":" + std::to_string(newReg+1) + "]";

          // Update read text
          size_t rp = r.text.find(oldS);
          if (rp != std::string::npos) r.text.replace(rp, oldS.size(), newS);

          // Update corresponding MFMA SrcA
          for (auto &m : groupMfmas[g]) {
            if (m.srcAReg == r.destReg) {
              size_t fv = m.text.find("v[");
              if (fv != std::string::npos) {
                size_t sv = m.text.find(oldS, fv + 1);
                if (sv != std::string::npos) {
                  m.text.replace(sv, oldS.size(), newS);
                  m.srcAReg = newReg;
                }
              }
              break; // only update one MFMA per read
            }
          }
          r.destReg = newReg;
        }
      }
    }

    // Classify filler instructions (everything that's not ds_read, mfma, or lgkmcnt)
    // Split into categories for interleaving between groups
    std::set<int> dsReadIdxs, mfmaIdxs;
    for (auto &r : reads) dsReadIdxs.insert(r.idx);
    for (auto &m : mfmas) mfmaIdxs.insert(m.idx);

    // Collect filler in original order, classifying by pattern
    struct Filler { std::string text; int cat; };
    // cat: 0=mask(v_cmp/v_cndmask), 1=rowmax(v_max3 before permute), 2=ds_permute,
    //      3=softmax(after permute), 4=pkfma, 5=buf_load(non-lds), 6=buf_load_lds_block,
    //      7=addr/salu, 8=vmcnt
    std::vector<Filler> fillers;
    bool seenPermute = false;
    bool seenPkFma = false;
    int max3Count = 0;
    for (int i = 0; i < (int)allLines.size(); i++) {
      if (dsReadIdxs.count(i) || mfmaIdxs.count(i) || allLines[i].type == 2) continue;
      std::string &t = allLines[i].text;
      std::string tl = t;
      for (auto &c : tl) c = tolower(c);
      int cat = 7; // default: addr/salu
      if (tl.find("v_cmp_") != std::string::npos || tl.find("v_cndmask") != std::string::npos)
        cat = 0;
      else if (tl.find("v_max3_f32") != std::string::npos) {
        max3Count++;
        if (!seenPermute) cat = 1; else cat = 3;
      }
      else if (tl.find("ds_permute") != std::string::npos) { cat = 2; seenPermute = true; }
      else if (tl.find("v_sub_f32") != std::string::npos || tl.find("v_mul_f32") != std::string::npos ||
               tl.find("v_max_f32") != std::string::npos || tl.find("v_min_f32") != std::string::npos) {
        if (seenPermute && !seenPkFma) cat = 3;
      }
      else if (tl.find("v_pk_fma") != std::string::npos) { cat = 4; seenPkFma = true; }
      else if (tl.find("buffer_load") != std::string::npos && tl.find("lds") != std::string::npos) cat = 6;
      else if (tl.find("buffer_load") != std::string::npos && tl.find("lds") == std::string::npos) cat = 5;
      else if (tl.find("s_waitcnt vmcnt") != std::string::npos) cat = 8;
      fillers.push_back({t, cat});
    }

    // Retroactively mark s_mov/s_nop/v_add adjacent to buffer_load_lds as cat=6
    // to keep the entire buffer_load_lds block together (m0 + offset + lds load)
    for (int i = 0; i < (int)fillers.size(); i++) {
      if (fillers[i].cat == 6) {
        // Mark preceding s_mov/s_nop/v_add as cat=6
        for (int j = i - 1; j >= 0; j--) {
          std::string tl = fillers[j].text;
          for (auto &c : tl) c = tolower(c);
          if (tl.find("s_mov_b32 m0") != std::string::npos ||
              tl.find("s_nop") != std::string::npos ||
              tl.find("v_add_u32") != std::string::npos) {
            fillers[j].cat = 6;
          } else {
            break;
          }
        }
      }
    }

    // Collect filler by category
    std::vector<std::string> maskFiller, rowmaxFiller, permuteFiller, softmaxFiller;
    std::vector<std::string> pkfmaFiller, bufLoadFiller, bufLdsFiller, addrFiller, vmcntFiller;
    for (auto &f : fillers) {
      switch (f.cat) {
        case 0: maskFiller.push_back(f.text); break;
        case 1: rowmaxFiller.push_back(f.text); break;
        case 2: permuteFiller.push_back(f.text); break;
        case 3: softmaxFiller.push_back(f.text); break;
        case 4: pkfmaFiller.push_back(f.text); break;
        case 5: bufLoadFiller.push_back(f.text); break;
        case 6: bufLdsFiller.push_back(f.text); break;
        case 7: addrFiller.push_back(f.text); break;
        case 8: vmcntFiller.push_back(f.text); break;
      }
    }

    // Build restructured region
    std::string newRegion;
    auto emitLines = [&](const std::vector<std::string> &lines) {
      for (auto &l : lines) newRegion += l + "\n";
    };
    auto emitReads = [&](int g) {
      for (auto &r : groupReads[g]) newRegion += r.text + "\n";
    };
    auto emitMfmas = [&](int g) {
      for (auto &m : groupMfmas[g]) newRegion += m.text + "\n";
    };

    // Group 0: reads → mask filler → lgkmcnt(0) → MFMAs
    emitReads(0);
    emitLines(maskFiller);
    emitLines(addrFiller);
    newRegion += "\ts_waitcnt lgkmcnt(0)\n";
    emitMfmas(0);

    // Group 1: reads → rowmax+permute → lgkmcnt(0) → softmax → MFMAs
    emitReads(1);
    emitLines(rowmaxFiller);
    emitLines(permuteFiller);
    newRegion += "\ts_waitcnt lgkmcnt(0)\n";
    emitLines(softmaxFiller);
    emitMfmas(1);

    // Group 2: reads → pkfma+vmcnt → lgkmcnt(0) → MFMAs
    emitReads(2);
    emitLines(vmcntFiller);
    emitLines(pkfmaFiller);
    newRegion += "\ts_waitcnt lgkmcnt(0)\n";
    emitMfmas(2);

    // Group 3: reads → buf_load → lgkmcnt(0) → MFMAs
    emitReads(3);
    emitLines(bufLoadFiller);
    newRegion += "\ts_waitcnt lgkmcnt(0)\n";
    emitMfmas(3);

    // Buffer_load_lds block at end (for next iteration's K data)
    emitLines(bufLdsFiller);

    loop.replace(regionStart, regionEnd - regionStart, newRegion);
    bodiesProcessed++;
    searchPos = regionStart + newRegion.size();
  }

  if (maxNewVgpr > maxNewVgprOut) maxNewVgprOut = maxNewVgpr;

  if (bodiesProcessed > 0)
    llvm::errs() << "[postProcessISA] restructureGEMM2_V3: processed "
                 << bodiesProcessed << " bodies, pool=" << maxNewVgpr
                 << " in " << label << "\n";
  return loop;
}

static std::string postProcessISA(const std::string &isa) {
  std::string result = isa;
  auto envEnabled = [](const char *name) {
    const char *v = std::getenv(name);
    if (!v)
      return false;
    std::string s(v);
    for (char &c : s)
      c = static_cast<char>(::tolower(c));
    return s == "1" || s == "true" || s == "yes" || s == "on";
  };
  const bool disableMaskSkip = envEnabled("FLIR_MASK_SKIP_DISABLE");
  const bool splitMaskSkipPass1Only =
      envEnabled("FLIR_MASK_SKIP_SPLIT_PASS1_ONLY");
  const bool debugMaskSkip = envEnabled("FLIR_MASK_SKIP_DEBUG");

  auto trimIsaLine = [](const std::string &line) -> std::string {
    size_t start = 0;
    while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
      start++;
    size_t end = line.size();
    while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
      end--;
    return line.substr(start, end - start);
  };

  auto parseLgkmCount = [&](const std::string &line) -> int {
    std::string t = trimIsaLine(line);
    size_t pos = t.find("lgkmcnt(");
    if (pos == std::string::npos)
      return -1;
    size_t numStart = pos + 8;
    size_t numEnd = t.find(')', numStart);
    if (numEnd == std::string::npos)
      return -1;
    return atoi(t.substr(numStart, numEnd - numStart).c_str());
  };

  auto splitIsaLines = [](const std::string &text) -> std::vector<std::string> {
    std::vector<std::string> lines;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
      lines.push_back(line);
    return lines;
  };

  auto isWaitRestoreAnchorSkippable = [&](const std::string &line) -> bool {
    return line.empty() || line == ";;#ASMSTART" || line == ";;#ASMEND" ||
           line.find("s_waitcnt") == 0 || line.find("s_barrier") == 0 ||
           line.find("s_setprio") == 0 || line.find("s_nop") == 0;
  };

  auto annotateWaitAnchorLines = [&](const std::vector<std::string> &lines,
                                     std::vector<std::string> &labels,
                                     std::vector<int> &mfmaOrdinals,
                                     std::vector<int> &dsReadOrdinals,
                                     std::vector<int> &vAddOrdinals) {
    labels.assign(lines.size(), "");
    mfmaOrdinals.assign(lines.size(), -1);
    dsReadOrdinals.assign(lines.size(), -1);
    vAddOrdinals.assign(lines.size(), -1);

    std::string currentLabel;
    std::map<std::string, int> nextMfmaOrdinal;
    std::map<std::string, int> nextDsReadOrdinal;
    std::map<std::string, int> nextVAddOrdinal;
    for (size_t i = 0; i < lines.size(); i++) {
      std::string t = trimIsaLine(lines[i]);
      if (!t.empty() && t[0] == '.' && t.back() == ':') {
        currentLabel = t.substr(0, t.size() - 1);
      }
      labels[i] = currentLabel;
      if (!currentLabel.empty() && t.find("v_mfma") == 0) {
        mfmaOrdinals[i] = nextMfmaOrdinal[currentLabel]++;
      }
      if (!currentLabel.empty() && t.find("ds_read") == 0) {
        dsReadOrdinals[i] = nextDsReadOrdinal[currentLabel]++;
      }
      if (!currentLabel.empty() && t.find("v_add_f32") == 0) {
        vAddOrdinals[i] = nextVAddOrdinal[currentLabel]++;
      }
    }
  };

  auto makeWaitRestoreKey = [&](const std::string &kind,
                                const std::string &label, int ordinal,
                                const std::string &anchor) -> std::string {
    return kind + "|" + label + "#" + std::to_string(ordinal) + "|" + anchor;
  };

  auto makeRelaxedWaitRestoreKey = [&](const std::string &kind,
                                       const std::string &label,
                                       const std::string &anchor)
      -> std::string { return kind + "|" + label + "|" + anchor; };

  auto buildOriginalLgkmWaitMap =
      [&](const std::string &text) -> std::map<std::string, unsigned> {
    std::vector<std::string> lines = splitIsaLines(text);
    std::vector<std::string> labels;
    std::vector<int> mfmaOrdinals;
    std::vector<int> dsReadOrdinals;
    std::vector<int> vAddOrdinals;
    annotateWaitAnchorLines(lines, labels, mfmaOrdinals, dsReadOrdinals,
                            vAddOrdinals);

    std::map<std::string, unsigned> waits;
    for (size_t i = 0; i < lines.size(); i++) {
      const int lgkmCnt = parseLgkmCount(lines[i]);
      if (lgkmCnt <= 0 || labels[i].empty())
        continue;

      for (size_t j = i + 1; j < lines.size() && j <= i + 32; j++) {
        if (labels[j] != labels[i])
          break;
        std::string anchor = trimIsaLine(lines[j]);
        if (isWaitRestoreAnchorSkippable(anchor))
          continue;

        if (anchor.find("ds_read") == 0 && dsReadOrdinals[j] >= 0) {
          waits.emplace(makeWaitRestoreKey("dsread", labels[j],
                                           dsReadOrdinals[j], anchor),
                        static_cast<unsigned>(lgkmCnt));
          waits.emplace(makeRelaxedWaitRestoreKey("dsread", labels[j], anchor),
                        static_cast<unsigned>(lgkmCnt));
          break;
        }
        if (anchor.find("v_add_f32") == 0 && vAddOrdinals[j] >= 0) {
          waits.emplace(makeWaitRestoreKey("vadd", labels[j], vAddOrdinals[j],
                                           anchor),
                        static_cast<unsigned>(lgkmCnt));
          waits.emplace(makeRelaxedWaitRestoreKey("vadd", labels[j], anchor),
                        static_cast<unsigned>(lgkmCnt));
          break;
        }
        if (anchor.find("v_mfma") == 0 && mfmaOrdinals[j] >= 0) {
          waits.emplace(makeWaitRestoreKey("mfma", labels[j], mfmaOrdinals[j],
                                           anchor),
                        static_cast<unsigned>(lgkmCnt));
          break;
        }
      }
    }
    return waits;
  };

  const std::map<std::string, unsigned> originalLgkmWaits =
      buildOriginalLgkmWaitMap(isa);
  const bool preserveBackendLgkmWaits = []() {
    const char *v = std::getenv("FLIR_PRESERVE_MFMA_LGKMCNT");
    if (!v)
      return true;
    std::string s(v);
    for (char &c : s)
      c = static_cast<char>(::tolower(c));
    return !(s == "0" || s == "false" || s == "no" || s == "off");
  }();

  // Find all inner loop labels with backward branch.
  // Detect both s_cbranch_vccnz and s_branch (unconditional) patterns,
  // since scf.if/else inside loops may change the backward branch type.
  std::vector<std::string> loopLabels;
  // Also track which branch pattern was used for each label
  std::map<std::string, std::string> loopBranchPattern;
  for (int fn = 0; fn <= 1; fn++) {
    for (int bb = 0; bb <= 30; bb++) {
      std::string label = ".LBB" + std::to_string(fn) + "_" + std::to_string(bb);
      std::string labelColon = label + ":";
      size_t labelPos = result.find(labelColon);
      if (labelPos == std::string::npos) continue;

      // Try multiple backward branch patterns
      std::vector<std::string> patterns = {
        "s_cbranch_vccnz " + label,
        "s_branch " + label,
      };
      for (const auto &pat : patterns) {
        size_t branchPos = result.find(pat, labelPos);
        if (branchPos != std::string::npos) {
          // Verify exact match (not prefix, e.g. .LBB0_5 vs .LBB0_50)
          size_t afterPat = branchPos + pat.size();
          if (afterPat >= result.size() || result[afterPat] == '\n' ||
              result[afterPat] == '\r') {
            loopLabels.push_back(label);
            loopBranchPattern[label] = pat;
            break;
          }
        }
      }
    }
  }

  auto countLoopBarriers = [&](const std::string &body) {
    unsigned count = 0;
    size_t bpos = 0;
    while (true) {
      size_t b = body.find("s_barrier", bpos);
      if (b == std::string::npos)
        break;
      count++;
      bpos = b + 10;
    }
    return count;
  };

  unsigned totalMultiBodyLoops = 0;
  for (const auto &label : loopLabels) {
    std::string labelColon = label + ":";
    std::string branchTarget = loopBranchPattern[label];
    size_t loopStart = result.find(labelColon);
    if (loopStart == std::string::npos)
      continue;
    size_t loopEnd = result.find(branchTarget, loopStart);
    if (loopEnd == std::string::npos)
      continue;
    size_t loopEndLine = result.find('\n', loopEnd);
    if (loopEndLine == std::string::npos)
      loopEndLine = result.size();
    std::string loop = result.substr(loopStart, loopEndLine - loopStart);
    if (countLoopBarriers(loop) > 2)
      totalMultiBodyLoops++;
  }

  unsigned multiBodyLoopOrdinal = 0;

  for (const auto &label : loopLabels) {
    std::string labelColon = label + ":";
    std::string branchTarget = loopBranchPattern[label];

    size_t loopStart = result.find(labelColon);
    if (loopStart == std::string::npos)
      continue;
    size_t loopEnd = result.find(branchTarget, loopStart);
    if (loopEnd == std::string::npos)
      continue;
    size_t loopEndLine = result.find('\n', loopEnd);
    if (loopEndLine == std::string::npos)
      loopEndLine = result.size();

    std::string before = result.substr(0, loopStart);
    std::string after = result.substr(loopEndLine);
    std::string loop = result.substr(loopStart, loopEndLine - loopStart);

    // Count barriers to detect 2x unrolled loops (>2 barriers = multi-body)
    unsigned loopBarriers = countLoopBarriers(loop);
    bool doYield = (loopBarriers > 2);
    bool fillGap = false;
    bool hoistVcmp = (loopBarriers <= 2);
    loop = processOneLoop(std::move(loop), label, doYield, fillGap, hoistVcmp);

    bool allowMaskSkipThisLoop = !disableMaskSkip;
    if (loopBarriers > 2) {
      if (splitMaskSkipPass1Only && totalMultiBodyLoops > 0) {
        unsigned pass1Start = totalMultiBodyLoops / 2;
        allowMaskSkipThisLoop = allowMaskSkipThisLoop &&
                                (multiBodyLoopOrdinal >= pass1Start);
        llvm::errs() << "[postProcessISA] mask-skip split-pass1-only: "
                     << (allowMaskSkipThisLoop ? "enable " : "disable ")
                     << label << " ordinal=" << multiBodyLoopOrdinal << "/"
                     << totalMultiBodyLoops << "\n";
      }
      multiBodyLoopOrdinal++;
    }

    // Pass 7: Move s_setprio around O rescale v_pk_mul block.
    // Compiler places s_setprio AFTER v_pk_mul; we move it BEFORE.
    {
      size_t lastPkMul = loop.rfind("v_pk_mul_f32");
      if (lastPkMul != std::string::npos) {
        // Find end of last v_pk_mul line
        size_t lastPkMulLineEnd = loop.find('\n', lastPkMul);
        if (lastPkMulLineEnd == std::string::npos)
          lastPkMulLineEnd = loop.size();
        else
          lastPkMulLineEnd++;

        // Scan backward to find the start of contiguous v_pk_mul block
        size_t blockStart = loop.rfind('\n', lastPkMul);
        if (blockStart == std::string::npos) blockStart = 0;
        else blockStart++;

        unsigned pkMulCount = 1;
        size_t scanPos = blockStart;
        while (scanPos > 0) {
          size_t prevLineEnd = scanPos - 1;
          size_t prevLineStart = loop.rfind('\n', prevLineEnd - 1);
          if (prevLineStart == std::string::npos) prevLineStart = 0;
          else prevLineStart++;
          std::string prevLine = loop.substr(prevLineStart,
                                             prevLineEnd - prevLineStart);
          if (prevLine.find("v_pk_mul_f32") != std::string::npos) {
            blockStart = prevLineStart;
            pkMulCount++;
            scanPos = prevLineStart;
          } else {
            break;
          }
        }

        if (false && pkMulCount >= 999) {
          // Look for s_setprio 1 / s_setprio 0 pair after the block
          std::string afterBlock = loop.substr(lastPkMulLineEnd,
              std::min((size_t)300, loop.size() - lastPkMulLineEnd));
          size_t sp1Off = afterBlock.find("s_setprio 1");
          size_t sp0Off = (sp1Off != std::string::npos)
                              ? afterBlock.find("s_setprio 0", sp1Off)
                              : std::string::npos;

          if (sp1Off != std::string::npos && sp0Off != std::string::npos
              && sp0Off < 200) {
            // Remove the pair (both lines)
            size_t sp1Abs = lastPkMulLineEnd + sp1Off;
            size_t sp1LineStart = loop.rfind('\n', sp1Abs);
            sp1LineStart = (sp1LineStart == std::string::npos) ? sp1Abs
                                                               : sp1LineStart + 1;
            size_t sp0Abs = lastPkMulLineEnd + sp0Off;
            size_t sp0LineEnd = loop.find('\n', sp0Abs);
            sp0LineEnd = (sp0LineEnd == std::string::npos) ? loop.size()
                                                           : sp0LineEnd + 1;
            loop.erase(sp1LineStart, sp0LineEnd - sp1LineStart);

            // Recalculate lastPkMulLineEnd after erasure
            lastPkMul = loop.rfind("v_pk_mul_f32");
            if (lastPkMul != std::string::npos) {
              lastPkMulLineEnd = loop.find('\n', lastPkMul);
              if (lastPkMulLineEnd != std::string::npos) lastPkMulLineEnd++;
              else lastPkMulLineEnd = loop.size();
            }

            // Insert s_setprio 0 after block
            loop.insert(lastPkMulLineEnd, "\ts_setprio 0\n");
            // Insert s_setprio 1 before block
            loop.insert(blockStart, "\ts_setprio 1\n");

            llvm::errs() << "[postProcessISA] Moved s_setprio around "
                         << pkMulCount << " v_pk_mul in " << label << "\n";
          }
        }
      }
    }

    // Pass 7.5: DISABLED — Yield windows after s_setprio 0 regress
    // (109T with 24cy, neutral with 4cy). MEM phase not separated from
    // COMPUTE, so yield doesn't create overlap.

    // Pass 8: DISABLED — V4 [4,4,4,4] grouping regresses to 106T.
    // With 4 SrcC groups, lgkmcnt stalls + s_nop padding (95 cy/body) exceed
    // SrcC transition savings (27 cy/body). Need ≤2 SrcC groups to benefit.
    // loop = restructureGEMM2_V4(std::move(loop), label);

    // Pass 9: DISABLED — V3 processes already-grouped Body A GEMM2 and breaks it.
    //          The interleaved Body 2/5 sections are the real target but V3 skips them.
    // static int g_maxNewVgpr = 0;
    // loop = restructureGEMM2_V3(std::move(loop), label, g_maxNewVgpr);

    if (allowMaskSkipThisLoop) {
      // --- Pass 10: Scalar branch to skip causal mask for non-boundary blocks ---
      // Generalized: matches ANY SGPR pair (not just s[0:1]), handles up to 2
      // bodies per loop. Two sub-cases:
      //   A) Tight v_cmp+v_cndmask block (cndCount >= 14): skip entire block
      //   B) v_cmp-only block with scattered v_cndmask: skip v_cmp, set SGPRs
      //      to exec so later v_cndmask passes through scores unchanged
      unsigned maskBranchesAdded = 0;
      if (loopBarriers > 2) {
      size_t mSearchPos = 0;
      while (maskBranchesAdded < 2) {
        // Find start of ANY mask block: v_cmp_lt_i32_e64 s[
        size_t firstCmp = loop.find("v_cmp_lt_i32_e64 s[", mSearchPos);
        if (firstCmp == std::string::npos) break;

        size_t firstCmpLineStart = loop.rfind('\n', firstCmp);
        firstCmpLineStart = (firstCmpLineStart == std::string::npos)
                                ? 0 : firstCmpLineStart + 1;
        size_t firstCmpLineEnd = loop.find('\n', firstCmp);
        if (firstCmpLineEnd == std::string::npos) {
          mSearchPos = firstCmp + 1; continue;
        }

        // Extract mask_delta register (last token)
        std::string cmpLine = loop.substr(firstCmp, firstCmpLineEnd - firstCmp);
        size_t lastComma = cmpLine.rfind(", ");
        if (lastComma == std::string::npos) {
          mSearchPos = firstCmpLineEnd + 1; continue;
        }
        std::string maskDeltaReg = cmpLine.substr(lastComma + 2);
        while (!maskDeltaReg.empty() && isspace(maskDeltaReg.back()))
          maskDeltaReg.pop_back();
        if (maskDeltaReg.empty() || maskDeltaReg[0] != 'v') {
          mSearchPos = firstCmpLineEnd + 1; continue;
        }

        // Count v_cmp_lt_i32 with maskDeltaReg, collect SGPR pairs, find max threshold
        int cmpCount = 0;
        int maxThreshold = -100;
        std::vector<std::string> sgprPairs;
        // Scattered single-path mask regions can stretch well beyond the old
        // 1500-char window once compares are interleaved with MFMA/load ops.
        size_t scanLim = std::min(firstCmpLineStart + 3000, loop.size());
        size_t pos = firstCmpLineStart;
        size_t lastCmpLineEnd = firstCmpLineEnd + 1;
        while (pos < scanLim) {
          size_t le = loop.find('\n', pos);
          if (le == std::string::npos) break;
          std::string cl = loop.substr(pos, le - pos);
          if (cl.find("v_cmp_lt_i32") != std::string::npos &&
              cl.find(maskDeltaReg) != std::string::npos) {
            cmpCount++;
            lastCmpLineEnd = le + 1;
            size_t c1 = cl.find(", ");
            if (c1 != std::string::npos) {
              size_t thS = c1 + 2;
              size_t thE = cl.find(",", thS);
              if (thE != std::string::npos) {
                std::string thStr = cl.substr(thS, thE - thS);
                int th = atoi(thStr.c_str());
                if (th > maxThreshold) maxThreshold = th;
              }
            }
            // Extract SGPR pair: "v_cmp_lt_i32_e64 s[N:M], ..."
            size_t sB = cl.find("s[");
            size_t sE = cl.find("]", sB);
            if (sB != std::string::npos && sE != std::string::npos) {
              std::string sp = cl.substr(sB, sE - sB + 1);
              bool found = false;
              for (const auto &p : sgprPairs) if (p == sp) { found = true; break; }
              if (!found) sgprPairs.push_back(sp);
            }
          }
          if (cl.find("s_barrier") != std::string::npos) break;
          pos = le + 1;
        }

        if (cmpCount < 14 || maxThreshold < 20 || sgprPairs.empty()) {
          if (debugMaskSkip) {
            llvm::errs() << "[postProcessISA] mask-skip debug: " << label
                         << " reject-cmp"
                         << " cmp=" << cmpCount
                         << " maxTh=" << maxThreshold
                         << " sgprs=" << sgprPairs.size()
                         << " delta=" << maskDeltaReg << "\n";
          }
          mSearchPos = firstCmpLineEnd + 1; continue;
        }

        // Use first SGPR pair from the mask block for the check (avoids
        // corrupting live SGPRs like s[0:1] which may hold kernel arg pointer)
        std::string checkSgpr = sgprPairs[0];

        // Collect v_max3_f32 instructions in the mask+rowmax block.
        std::vector<std::string> max3Lines;
        size_t lastMax3End = 0;
        pos = firstCmpLineStart;
        scanLim = std::min(firstCmpLineStart + 2500, loop.size());
        while (pos < scanLim) {
          size_t le = loop.find('\n', pos);
          if (le == std::string::npos) break;
          std::string cl = loop.substr(pos, le - pos);
          if (cl.find("v_max3_f32") != std::string::npos &&
              cl.find(maskDeltaReg) != std::string::npos) {
            max3Lines.push_back(cl);
            lastMax3End = le + 1;
          }
          if (cl.find("s_waitcnt") != std::string::npos ||
              cl.find("ds_permute") != std::string::npos) break;
          pos = le + 1;
        }

        if (max3Lines.size() < 7) {
          // Current single-path kernel shape: v_cmp/v_cndmask are scattered across
          // a region interleaved with MFMAs, LDS reads, and buffer_loads.
          // Build a fast path by cloning that whole region, removing only the
          // compares, and pre-setting the mask SGPR pairs to exec so the later
          // v_cndmask instructions become pass-through on non-boundary blocks.
          {
            size_t regionEnd = 0;
            int regionCmpCount = 0;
            int regionCndCount = 0;
            int regionMax3Count = 0;
            bool seenDsPermute = false;
            bool unsafeBranchUse = false;
            std::vector<std::string> regionLines;

            auto isScatteredMaskLine = [&](const std::string &cl) {
              std::string tl = cl;
              for (auto &c : tl) c = tolower(c);
              return (tl.find("v_cmp_") != std::string::npos &&
                      tl.find(maskDeltaReg) != std::string::npos) ||
                     tl.find("v_cndmask_b32") != std::string::npos ||
                     tl.find("v_mfma") != std::string::npos ||
                     tl.find("ds_read") != std::string::npos ||
                     tl.find("buffer_load") != std::string::npos ||
                     tl.find("s_mov_b32 m0") != std::string::npos ||
                     tl.find("s_waitcnt") != std::string::npos ||
                     tl.find("v_add_u32") != std::string::npos ||
                     tl.find("s_setprio") != std::string::npos ||
                     tl.find("v_max3_f32") != std::string::npos ||
                     tl.find("ds_permute") != std::string::npos;
            };

            pos = firstCmpLineStart;
            scanLim = std::min(firstCmpLineStart + 3500, loop.size());
            while (pos < scanLim) {
              size_t le = loop.find('\n', pos);
              if (le == std::string::npos) break;
              std::string cl = loop.substr(pos, le - pos);
              std::string tl = cl;
              for (auto &c : tl) c = tolower(c);

              bool isCmp = (tl.find("v_cmp_") != std::string::npos &&
                            tl.find(maskDeltaReg) != std::string::npos);
              bool isAllowed = isScatteredMaskLine(cl);

              if (tl.find("s_cbranch_vcc") != std::string::npos ||
                  tl.find("s_cbranch_exec") != std::string::npos ||
                  tl.find("s_and_saveexec") != std::string::npos ||
                  tl.find("s_andn2_saveexec") != std::string::npos)
                unsafeBranchUse = true;

              if (isCmp) regionCmpCount++;
              if (tl.find("v_cndmask_b32") != std::string::npos) regionCndCount++;
              if (tl.find("v_max3_f32") != std::string::npos) regionMax3Count++;
              if (tl.find("ds_permute") != std::string::npos) seenDsPermute = true;

              if (isAllowed) {
                regionLines.push_back(cl);
                regionEnd = le + 1;
                if (seenDsPermute && tl.find("s_waitcnt lgkmcnt(0)") != std::string::npos)
                  break;
                pos = le + 1;
                continue;
              }

              if (regionCmpCount > 0 &&
                  !cl.empty() && cl.find_first_not_of(" \t") != std::string::npos)
                break;
              pos = le + 1;
            }

            if (!unsafeBranchUse &&
                regionCmpCount >= 14 &&
                regionCndCount >= 12 &&
                regionMax3Count >= 7 &&
                seenDsPermute &&
                regionEnd != 0) {
              std::string fastLbl = ".Lfm_" + label.substr(1) + "_" +
                                    std::to_string(maskBranchesAdded);
              std::string mergeLbl = ".Lmm_" + label.substr(1) + "_" +
                                     std::to_string(maskBranchesAdded);

              std::string checkCode;
              checkCode += "\tv_cmp_ge_i32_e64 " + checkSgpr + ", " +
                           std::to_string(maxThreshold) + ", " +
                           maskDeltaReg + "\n";
              checkCode += "\ts_and_b64 " + checkSgpr + ", " + checkSgpr + ", exec\n";
              checkCode += "\ts_cbranch_scc0 " + fastLbl + "\n";

              loop.insert(firstCmpLineStart, checkCode);
              size_t checkLen = checkCode.size();
              regionEnd += checkLen;

              std::string fastCode;
              fastCode += "\ts_branch " + mergeLbl + "\n";
              fastCode += fastLbl + ":\n";
              for (const auto &sp : sgprPairs)
                fastCode += "\ts_mov_b64 " + sp + ", exec\n";
              for (const auto &rl : regionLines) {
                std::string tl = rl;
                for (auto &c : tl) c = tolower(c);
                if (tl.find("v_cmp_") != std::string::npos) continue;
                fastCode += rl + "\n";
              }
              fastCode += mergeLbl + ":\n";
              loop.insert(regionEnd, fastCode);

              llvm::errs() << "[postProcessISA] Added mask skip (scattered-region) in "
                           << label
                           << " body " << maskBranchesAdded
                           << " (cmp=" << regionCmpCount
                           << ", cnd=" << regionCndCount
                           << ", max3=" << regionMax3Count
                           << ", maxTh=" << maxThreshold
                           << ", delta=" << maskDeltaReg << ")\n";

              maskBranchesAdded++;
              mSearchPos = regionEnd + fastCode.size();
              continue;
            }
            if (debugMaskSkip) {
              llvm::errs() << "[postProcessISA] mask-skip debug: " << label
                           << " reject-scattered"
                           << " cmp=" << regionCmpCount
                           << " cnd=" << regionCndCount
                           << " max3=" << regionMax3Count
                           << " dspermute=" << seenDsPermute
                           << " unsafe=" << unsafeBranchUse
                           << " regionEnd=" << (regionEnd != 0)
                           << " delta=" << maskDeltaReg << "\n";
            }
          }

          // No interleaved v_max3. Try two sub-cases:
          // (A) Tight block: v_cmp + v_cndmask all together (cndCount >= 14)
          // (B) Scattered: v_cmp block only, v_cndmask after MFMAs
          size_t lastCndEnd = 0;
          int cndCount = 0;
          std::vector<std::string> sideEffectLines;
          pos = firstCmpLineStart;
          scanLim = std::min(firstCmpLineStart + 2500, loop.size());
          while (pos < scanLim) {
            size_t le = loop.find('\n', pos);
            if (le == std::string::npos) break;
            std::string cl = loop.substr(pos, le - pos);
            if (cl.find("v_cndmask_b32_e64") != std::string::npos ||
                cl.find("v_cndmask_b32_e32") != std::string::npos) {
              cndCount++;
              lastCndEnd = le + 1;
            } else if (cl.find("v_cmp_lt_i32") == std::string::npos &&
                       cl.find("v_cmp_ge_i32") == std::string::npos &&
                       !cl.empty() && cl.find_first_not_of(" \t") != std::string::npos) {
              if (cl.find("buffer_load") != std::string::npos ||
                  cl.find("ds_read") != std::string::npos ||
                  cl.find("s_mov") != std::string::npos ||
                  cl.find("s_waitcnt") != std::string::npos) {
                sideEffectLines.push_back(cl);
              }
            }
            if (cndCount > 0 && cl.find("v_cndmask") == std::string::npos &&
                cl.find("v_cmp_lt") == std::string::npos &&
                cl.find("v_cmp_ge") == std::string::npos &&
                cl.find("buffer_load") == std::string::npos &&
                cl.find("ds_read") == std::string::npos &&
                cl.find("s_mov") == std::string::npos &&
                cl.find("s_waitcnt") == std::string::npos &&
                !cl.empty() && cl.find_first_not_of(" \t") != std::string::npos) {
              break;
            }
            pos = le + 1;
          }

          std::string fastLbl = ".Lfm_" + label.substr(1) + "_" +
                                std::to_string(maskBranchesAdded);
          std::string mergeLbl = ".Lmm_" + label.substr(1) + "_" +
                                 std::to_string(maskBranchesAdded);

          if (cndCount >= 14 && lastCndEnd != 0) {
            // Case A: tight block — skip entire v_cmp+v_cndmask region
            std::string checkCode;
            checkCode += "\tv_cmp_ge_i32_e64 " + checkSgpr + ", " +
                         std::to_string(maxThreshold) + ", " +
                         maskDeltaReg + "\n";
            checkCode += "\ts_and_b64 " + checkSgpr + ", " + checkSgpr + ", exec\n";
            checkCode += "\ts_cbranch_scc0 " + fastLbl + "\n";

            loop.insert(firstCmpLineStart, checkCode);
            size_t checkLen = checkCode.size();
            lastCndEnd += checkLen;

            std::string fastCode;
            fastCode += "\ts_branch " + mergeLbl + "\n";
            fastCode += fastLbl + ":\n";
            for (const auto &sl : sideEffectLines)
              fastCode += sl + "\n";
            fastCode += mergeLbl + ":\n";
            loop.insert(lastCndEnd, fastCode);

            llvm::errs() << "[postProcessISA] Added mask skip (tight) in " << label
                         << " body " << maskBranchesAdded
                         << " (cmp=" << cmpCount
                         << ", cnd=" << cndCount
                         << ", maxTh=" << maxThreshold
                         << ", delta=" << maskDeltaReg << ")\n";

            maskBranchesAdded++;
            mSearchPos = lastCndEnd + fastCode.size();
            continue;
          }

          // Case B: scattered v_cndmask — skip v_cmp block only, set SGPRs to exec
          // The later v_cndmask in the GEMM2 region will use exec predicates
          // and pass through scores unchanged (correct for non-boundary blocks).
          {
            // Rebuild the compare block in-place, replacing each mask compare with
            // the corresponding exec predicate write at the same position. This
            // preserves scalar-address lifetimes like s0/s4 that are still used by
            // buffer_load instructions before their final compare redefines them.
            // The companion FlyDSL K-soffset load form depends on that ordering.
            std::vector<std::string> cmpFastLines;
            pos = firstCmpLineStart;
            while (pos < lastCmpLineEnd) {
              size_t le = loop.find('\n', pos);
              if (le == std::string::npos || le >= lastCmpLineEnd) break;
              std::string cl = loop.substr(pos, le - pos);
              bool isMaskCmp =
                  (cl.find("v_cmp_") != std::string::npos &&
                   cl.find(maskDeltaReg) != std::string::npos);
              if (isMaskCmp) {
                if (cl.find("vcc") != std::string::npos) {
                  cmpFastLines.push_back("\ts_mov_b64 vcc, exec");
                } else {
                  size_t sB = cl.find("s[");
                  size_t sE = cl.find("]", sB);
                  if (sB != std::string::npos && sE != std::string::npos) {
                    std::string sp = cl.substr(sB, sE - sB + 1);
                    cmpFastLines.push_back("\ts_mov_b64 " + sp + ", exec");
                  }
                }
              } else if (!cl.empty() &&
                         cl.find_first_not_of(" \t") != std::string::npos) {
                cmpFastLines.push_back(cl);
              }
              pos = le + 1;
            }

            std::string checkCode;
            checkCode += "\tv_cmp_ge_i32_e64 " + checkSgpr + ", " +
                         std::to_string(maxThreshold) + ", " +
                         maskDeltaReg + "\n";
            checkCode += "\ts_and_b64 " + checkSgpr + ", " + checkSgpr + ", exec\n";
            checkCode += "\ts_cbranch_scc0 " + fastLbl + "\n";

            loop.insert(firstCmpLineStart, checkCode);
            size_t checkLen = checkCode.size();
            lastCmpLineEnd += checkLen;

            std::string fastCode;
            fastCode += "\ts_branch " + mergeLbl + "\n";
            fastCode += fastLbl + ":\n";
            for (const auto &sl : cmpFastLines)
              fastCode += sl + "\n";
            fastCode += mergeLbl + ":\n";
            loop.insert(lastCmpLineEnd, fastCode);

            llvm::errs() << "[postProcessISA] Added mask skip (sgpr-set) in " << label
                         << " body " << maskBranchesAdded
                         << " (cmp=" << cmpCount
                         << ", sgprs=" << sgprPairs.size()
                         << ", maxTh=" << maxThreshold
                         << ", delta=" << maskDeltaReg << ")\n";

            maskBranchesAdded++;
            mSearchPos = lastCmpLineEnd + fastCode.size();
            continue;
          }
        }

        // Main path: v_max3 interleaved with v_cmp+v_cndmask
        size_t firstMax3Pos = loop.find("v_max3_f32", lastCmpLineEnd);
        if (firstMax3Pos == std::string::npos ||
            firstMax3Pos - lastCmpLineEnd > 600) {
          if (debugMaskSkip) {
            llvm::errs() << "[postProcessISA] mask-skip debug: " << label
                         << " reject-max3-gap"
                         << " cmp=" << cmpCount
                         << " max3gap="
                         << ((firstMax3Pos == std::string::npos)
                                 ? -1
                                 : static_cast<int>(firstMax3Pos - lastCmpLineEnd))
                         << " delta=" << maskDeltaReg << "\n";
          }
          mSearchPos = firstCmpLineEnd + 1; continue;
        }
        if (lastMax3End == 0 || max3Lines.empty()) {
          if (debugMaskSkip) {
            llvm::errs() << "[postProcessISA] mask-skip debug: " << label
                         << " reject-max3-block"
                         << " cmp=" << cmpCount
                         << " max3=" << max3Lines.size()
                         << " delta=" << maskDeltaReg << "\n";
          }
          mSearchPos = firstCmpLineEnd + 1; continue;
        }

        std::string fastLbl = ".Lfm_" + label.substr(1) + "_" +
                              std::to_string(maskBranchesAdded);
        std::string mergeLbl = ".Lmm_" + label.substr(1) + "_" +
                               std::to_string(maskBranchesAdded);

        std::string checkCode;
        checkCode += "\tv_cmp_ge_i32_e64 " + checkSgpr + ", " +
                     std::to_string(maxThreshold) + ", " +
                     maskDeltaReg + "\n";
        checkCode += "\ts_and_b64 " + checkSgpr + ", " + checkSgpr + ", exec\n";
        checkCode += "\ts_cbranch_scc0 " + fastLbl + "\n";

        loop.insert(firstCmpLineStart, checkCode);
        size_t checkLen = checkCode.size();
        lastMax3End += checkLen;

        std::string fastCode;
        fastCode += "\ts_branch " + mergeLbl + "\n";
        fastCode += fastLbl + ":\n";
        for (const auto &ml : max3Lines)
          fastCode += ml + "\n";
        fastCode += mergeLbl + ":\n";

        loop.insert(lastMax3End, fastCode);

        llvm::errs() << "[postProcessISA] Added mask skip branch in " << label
                     << " body " << maskBranchesAdded
                     << " (cmp=" << cmpCount
                     << ", maxTh=" << maxThreshold
                     << ", max3=" << max3Lines.size()
                     << ", delta=" << maskDeltaReg << ")\n";

        maskBranchesAdded++;
        mSearchPos = lastMax3End + fastCode.size();
      }

      // --- Pass 10b: Body B mask skip (v_cmp block only, set SGPRs to exec) ---
      // DISABLED: Body B v_cmp block often has buffer_load interleaved.
      // Duplicating them in the fast path causes memory faults in some loops.
      // Body B savings are small (~64 cycles of SALU v_cmp), not worth the risk.
      mSearchPos = 0;
      while (false) {
        std::string bodyBMarker = "v_cmp_lt_i32_e32 vcc, ";
        size_t vcmpPos = loop.find(bodyBMarker, mSearchPos);
        if (vcmpPos == std::string::npos) break;

        size_t vcmpLineStart = loop.rfind('\n', vcmpPos);
        vcmpLineStart = (vcmpLineStart == std::string::npos)
                            ? 0 : vcmpLineStart + 1;
        size_t vcmpLineEnd = loop.find('\n', vcmpPos);
        if (vcmpLineEnd == std::string::npos) {
          mSearchPos = vcmpPos + 1; continue;
        }

        // Extract mask_delta register
        std::string bLine = loop.substr(vcmpPos, vcmpLineEnd - vcmpPos);
        size_t bLastComma = bLine.rfind(", ");
        if (bLastComma == std::string::npos) {
          mSearchPos = vcmpLineEnd + 1; continue;
        }
        std::string bMaskReg = bLine.substr(bLastComma + 2);
        while (!bMaskReg.empty() && isspace(bMaskReg.back()))
          bMaskReg.pop_back();
        if (bMaskReg.empty() || bMaskReg[0] != 'v') {
          mSearchPos = vcmpLineEnd + 1; continue;
        }

        // Count v_cmp_lt_i32 in this block (tolerates interleaved loads/s_mov)
        int bCmpCount = 0;
        size_t bLastCmpEnd = vcmpLineEnd + 1;
        std::vector<std::string> sgprPairs;
        std::vector<std::string> bSideEffects;
        bool usesVcc = false;
        size_t bScanLim = std::min(vcmpLineStart + 1200, loop.size());
        size_t bPos = vcmpLineStart;
        while (bPos < bScanLim) {
          size_t ble = loop.find('\n', bPos);
          if (ble == std::string::npos) break;
          std::string bcl = loop.substr(bPos, ble - bPos);
          if (bcl.find("v_cmp_lt_i32") != std::string::npos &&
              bcl.find(bMaskReg) != std::string::npos) {
            bCmpCount++;
            bLastCmpEnd = ble + 1;
            if (bcl.find("vcc") != std::string::npos)
              usesVcc = true;
            size_t sPos = bcl.find("s[");
            if (sPos != std::string::npos) {
              size_t sEnd = bcl.find(']', sPos);
              if (sEnd != std::string::npos) {
                std::string sp = bcl.substr(sPos, sEnd - sPos + 1);
                bool found = false;
                for (const auto &existing : sgprPairs)
                  if (existing == sp) { found = true; break; }
                if (!found) sgprPairs.push_back(sp);
              }
            }
          } else if (bCmpCount > 0) {
            if (bcl.find("buffer_load") != std::string::npos ||
                bcl.find("ds_read") != std::string::npos ||
                bcl.find("s_mov") != std::string::npos ||
                bcl.find("s_waitcnt") != std::string::npos) {
              bSideEffects.push_back(bcl);
              bLastCmpEnd = ble + 1;
              bPos = ble + 1;
              continue;
            }
            break;
          }
          bPos = ble + 1;
        }

        if (bCmpCount < 14) {
          mSearchPos = vcmpLineEnd + 1; continue;
        }

        std::string bFastLbl = ".Lfm_" + label.substr(1) + "_" +
                               std::to_string(maskBranchesAdded);
        std::string bMergeLbl = ".Lmm_" + label.substr(1) + "_" +
                                std::to_string(maskBranchesAdded);

        // Insert check before the v_cmp block
        std::string bCheck;
        bCheck += "\tv_cmp_ge_i32_e64 s[0:1], 26, " + bMaskReg + "\n";
        bCheck += "\ts_and_b64 s[0:1], s[0:1], exec\n";
        bCheck += "\ts_cbranch_scc0 " + bFastLbl + "\n";
        loop.insert(vcmpLineStart, bCheck);
        bLastCmpEnd += bCheck.size();

        // Insert fast path: duplicate side-effect instructions + set preds to exec
        std::string bFast;
        bFast += "\ts_branch " + bMergeLbl + "\n";
        bFast += bFastLbl + ":\n";
        for (const auto &se : bSideEffects)
          bFast += se + "\n";
        if (usesVcc)
          bFast += "\ts_mov_b64 vcc, exec\n";
        for (const auto &sp : sgprPairs)
          bFast += "\ts_mov_b64 " + sp + ", exec\n";
        bFast += bMergeLbl + ":\n";
        loop.insert(bLastCmpEnd, bFast);

        llvm::errs() << "[postProcessISA] Added mask skip (Body B) in " << label
                     << " body " << maskBranchesAdded
                     << " (cmp=" << bCmpCount
                     << ", sgprPairs=" << sgprPairs.size()
                     << ", vcc=" << usesVcc
                     << ", delta=" << bMaskReg << ")\n";

        maskBranchesAdded++;
        mSearchPos = bLastCmpEnd + bFast.size();
      }
      }
    }

    // --- Pass 11: Relocate s_setprio around O rescale + add yield NOPs ---
    // DISABLED: Relocating s_setprio from GEMM1 boundary (compiler-chosen)
    // to O rescale boundary (reference ASM pattern) caused regression
    // (111T → 110.4T without yield, 107.7T with yield). The 2-body loop
    // structure has different timing than the reference's 5-body loop.

    // --- Pass 12: Add yield NOPs after s_setprio 0 ---
    // DISABLED: Yield NOPs (s_nop 15 + s_nop 7) after s_setprio 0 caused
    // regression (111T → 109T). The 2-body loop doesn't benefit from the
    // reference ASM's yield pattern.

    // --- Pass 13: Move s_setprio 1 right after "compute-start" barriers ---
    // DISABLED: Moving s_setprio 1 earlier causes waves to compete more
    // aggressively for shared resources, resulting in regression (112.5→111.4T).
    // The low-priority exp2 phase allows natural wave interleaving.
    if (false) {
      unsigned setprioMoved = 0;
      // Process each s_barrier in the loop
      size_t searchPos = 0;
      while (true) {
        size_t bar = loop.find("s_barrier", searchPos);
        if (bar == std::string::npos) break;
        size_t barLineEnd = loop.find('\n', bar);
        if (barLineEnd == std::string::npos) { searchPos = bar + 10; continue; }
        barLineEnd++;

        // Only process barriers where the first instruction starts with v_
        // (compute-start barriers). Skip V staging barriers (ds_write),
        // loop exit barriers (s_cbranch), and wait barriers (s_waitcnt).
        std::string afterBar = loop.substr(barLineEnd,
            std::min((size_t)80, loop.size() - barLineEnd));
        // Find first non-whitespace instruction
        size_t firstInst = afterBar.find_first_not_of(" \t\n");
        if (firstInst == std::string::npos ||
            afterBar.substr(firstInst, 2) != "v_") {
          searchPos = barLineEnd;
          continue;
        }

        // Find s_setprio 1 within the next 4000 chars
        size_t sp1 = loop.find("s_setprio 1", barLineEnd);
        if (sp1 == std::string::npos || sp1 - barLineEnd > 4000) {
          searchPos = barLineEnd;
          continue;
        }

        // Extract the s_setprio 1 block (including ASMSTART/ASMEND markers)
        size_t sp1LineStart = loop.rfind('\n', sp1);
        sp1LineStart = (sp1LineStart == std::string::npos) ? 0 : sp1LineStart + 1;
        size_t sp1LineEnd = loop.find('\n', sp1);
        sp1LineEnd = (sp1LineEnd == std::string::npos) ? loop.size() : sp1LineEnd + 1;

        // Look for ASMSTART before and ASMEND after
        size_t asmStart = sp1LineStart;
        if (sp1LineStart > 0) {
          std::string beforeSp1 = loop.substr(
              (sp1LineStart > 40) ? sp1LineStart - 40 : 0,
              (sp1LineStart > 40) ? 40 : sp1LineStart);
          size_t asmS = beforeSp1.rfind(";;#ASMSTART");
          if (asmS != std::string::npos) {
            size_t asmSLineStart = beforeSp1.rfind('\n', asmS);
            asmStart = ((sp1LineStart > 40) ? sp1LineStart - 40 : 0) +
                       ((asmSLineStart == std::string::npos) ? 0 : asmSLineStart + 1);
          }
        }
        size_t asmEnd = sp1LineEnd;
        if (sp1LineEnd < loop.size()) {
          std::string afterSp1 = loop.substr(sp1LineEnd,
              std::min((size_t)40, loop.size() - sp1LineEnd));
          size_t asmE = afterSp1.find(";;#ASMEND");
          if (asmE != std::string::npos) {
            size_t asmEEnd = afterSp1.find('\n', asmE);
            asmEnd = sp1LineEnd + ((asmEEnd == std::string::npos) ? afterSp1.size() : asmEEnd + 1);
          }
        }

        std::string sp1Block = loop.substr(asmStart, asmEnd - asmStart);

        // Only move if s_setprio 1 is actually AFTER the barrier (not already adjacent)
        if (asmStart <= barLineEnd + 5) {
          searchPos = asmEnd;
          continue;
        }

        // Remove from current position and insert right after barrier
        loop.erase(asmStart, asmEnd - asmStart);
        // Recalculate barLineEnd (it hasn't changed since erasure is after it)
        loop.insert(barLineEnd, sp1Block);
        setprioMoved++;
        llvm::errs() << "[postProcessISA] Moved s_setprio 1 to after barrier in "
                     << label << "\n";
        searchPos = barLineEnd + sp1Block.size();
      }
    }

    // --- Pass 14: Remove redundant pre-barrier lgkmcnt(0) ---
    // DISABLED: The lgkmcnt(0) before barriers serves as a useful scheduling
    // fence even when technically redundant. Removing it can regress.
    if (false) {
      unsigned redundantLgkm = 0;
      const std::string lgk0 = "s_waitcnt lgkmcnt(0)";
      size_t pos = 0;
      while (true) {
        size_t wc = loop.find(lgk0, pos);
        if (wc == std::string::npos) break;
        size_t wcLineEnd = loop.find('\n', wc);
        wcLineEnd = (wcLineEnd == std::string::npos) ? loop.size() : wcLineEnd + 1;

        // Check if followed by s_barrier within 5 lines
        std::string afterWc = loop.substr(wcLineEnd,
            std::min((size_t)60, loop.size() - wcLineEnd));
        if (afterWc.find("s_barrier") == std::string::npos) {
          pos = wcLineEnd;
          continue;
        }

        // Check if this lgkmcnt(0) is part of a FULL wait
        size_t wcLineStart = loop.rfind('\n', wc);
        wcLineStart = (wcLineStart == std::string::npos) ? 0 : wcLineStart + 1;
        std::string wcLine = loop.substr(wcLineStart, wcLineEnd - wcLineStart);
        if (wcLine.find("vmcnt") != std::string::npos ||
            wcLine.find("expcnt") != std::string::npos) {
          pos = wcLineEnd;
          continue;
        }

        // Look back for the previous lgkmcnt(0)
        size_t lookStart = (wcLineStart > 5000) ? wcLineStart - 5000 : 0;
        std::string before = loop.substr(lookStart, wcLineStart - lookStart);
        size_t prevLgkm = before.rfind("lgkmcnt(0)");
        if (prevLgkm == std::string::npos) {
          pos = wcLineEnd;
          continue;
        }

        // Check no ds_ operations between previous lgkmcnt(0) and this one
        std::string gap = before.substr(prevLgkm);
        bool hasDS = gap.find("ds_read") != std::string::npos ||
                     gap.find("ds_write") != std::string::npos ||
                     gap.find("ds_permute") != std::string::npos;
        if (!hasDS) {
          loop.erase(wcLineStart, wcLineEnd - wcLineStart);
          redundantLgkm++;
          llvm::errs() << "[postProcessISA] Removed redundant pre-barrier lgkmcnt(0) in "
                       << label << "\n";
          continue;
        }
        pos = wcLineEnd;
      }
    }

    // --- Pass 25: Move V reads into GEMM1 MFMA chain ---
    // GEMM1 MFMAs #1-4 use v[194:201] as SrcA (K data). After each MFMA
    // issues, those registers are free. V reads (ds_read_b64 to v[194:201])
    // can be interleaved with later GEMM1 MFMAs. LDS reads use a different
    // issue port than MFMA, so they're essentially free. And V data arrives
    // before GEMM2 starts, eliminating lgkmcnt stalls.
    {
      unsigned vreadsMoved = 0;

      // Find each GEMM1 chain: 16 consecutive v_mfma followed by s_setprio 0
      // The V reads are after s_setprio 0: ds_read_b64 × 4
      size_t sp0search = 0;
      while (true) {
        // Find s_setprio 0 (marks GEMM1 end)
        size_t sp0 = loop.find("s_setprio 0", sp0search);
        if (sp0 == std::string::npos) break;
        size_t sp0LineStart = loop.rfind('\n', sp0);
        sp0LineStart = (sp0LineStart == std::string::npos) ? 0 : sp0LineStart + 1;
        size_t sp0BlockEnd = loop.find('\n', sp0);
        if (sp0BlockEnd == std::string::npos) break;
        sp0BlockEnd++;
        // Include ASMEND line if present
        {
          size_t ae = loop.find(";;#ASMEND", sp0BlockEnd);
          if (ae != std::string::npos && ae - sp0BlockEnd < 20) {
            size_t aeEnd = loop.find('\n', ae);
            if (aeEnd != std::string::npos) sp0BlockEnd = aeEnd + 1;
          }
        }

        // Verify s_setprio 0 is preceded by v_mfma (GEMM1 end)
        size_t asmStartCheck = loop.rfind(";;#ASMSTART", sp0);
        size_t checkFrom = (asmStartCheck != std::string::npos && sp0 - asmStartCheck < 30)
            ? asmStartCheck : sp0;
        size_t lookBack = (checkFrom > 200) ? checkFrom - 200 : 0;
        std::string beforeSp0 = loop.substr(lookBack, checkFrom - lookBack);
        if (beforeSp0.rfind("v_mfma") == std::string::npos) {
          sp0search = sp0BlockEnd;
          continue;
        }

        // Collect ds_read_b64 lines after s_setprio 0 (the V reads)
        std::vector<std::string> vreadLines;
        std::vector<std::pair<size_t,size_t>> vreadRanges;
        size_t scanPos = sp0BlockEnd;
        for (int i = 0; i < 10 && scanPos < loop.size(); i++) {
          size_t lineEnd = loop.find('\n', scanPos);
          if (lineEnd == std::string::npos) break;
          std::string line = loop.substr(scanPos, lineEnd - scanPos);
          // Trim leading whitespace for check
          std::string trimmed = line;
          while (!trimmed.empty() && (trimmed[0] == ' ' || trimmed[0] == '\t'))
            trimmed.erase(0, 1);

          if (trimmed.find("ds_read_b64") == 0) {
            vreadLines.push_back(line);
            vreadRanges.push_back({scanPos, lineEnd + 1});
            scanPos = lineEnd + 1;
          } else if (trimmed.empty()) {
            scanPos = lineEnd + 1;
          } else {
            break;
          }
        }

        if (vreadLines.size() < 4) {
          sp0search = sp0BlockEnd;
          continue;
        }

        // Only move the first 4 V reads
        if (vreadLines.size() > 4) {
          vreadLines.resize(4);
          vreadRanges.resize(4);
        }

        // Find the GEMM1 MFMA chain (16 v_mfma before s_setprio 0)
        // Walk backward from s_setprio 0 to find all MFMAs
        std::vector<std::pair<size_t,size_t>> mfmaRanges;
        {
          size_t mfmaSearchEnd = sp0LineStart;
          // Also check if there's ASMSTART before s_setprio
          if (asmStartCheck != std::string::npos && sp0 - asmStartCheck < 30) {
            size_t asmStartLine = loop.rfind('\n', asmStartCheck);
            if (asmStartLine != std::string::npos)
              mfmaSearchEnd = asmStartLine + 1;
          }

          size_t cur = mfmaSearchEnd;
          while (mfmaRanges.size() < 16 && cur > 0) {
            cur--;
            size_t lineStart = loop.rfind('\n', cur);
            lineStart = (lineStart == std::string::npos) ? 0 : lineStart + 1;
            std::string line = loop.substr(lineStart, cur + 1 - lineStart);
            // Trim
            std::string trimmed = line;
            while (!trimmed.empty() && (trimmed[0] == ' ' || trimmed[0] == '\t'))
              trimmed.erase(0, 1);
            if (trimmed.find("v_mfma") == 0) {
              mfmaRanges.push_back({lineStart, cur + 2}); // +2 for newline
              cur = lineStart;
              if (lineStart > 0) cur--;
            } else if (trimmed.empty()) {
              cur = lineStart;
              if (lineStart > 0) cur--;
            } else {
              break;
            }
          }
        }

        // Reverse to get in order (we collected backward)
        std::reverse(mfmaRanges.begin(), mfmaRanges.end());

        if (mfmaRanges.size() < 16) {
          sp0search = sp0BlockEnd;
          continue;
        }

        // Strategy: insert V reads after MFMA #4, #6, #8, #10
        // This spaces them by ~8 cycles (2 MFMAs × 4cy) for LDS port spacing
        // V reads go after MFMAs [4, 6, 8, 10] (0-indexed: [3, 5, 7, 9])
        int insertPoints[] = {4, 6, 8, 10}; // 1-indexed MFMA numbers

        // Remove original V reads first (in reverse order)
        for (int i = (int)vreadRanges.size() - 1; i >= 0; i--) {
          loop.erase(vreadRanges[i].first,
                     vreadRanges[i].second - vreadRanges[i].first);
        }

        // Recalculate MFMA positions after erasure
        // The erasure was AFTER the MFMAs, so positions are unchanged.

        // Re-find the MFMA chain (positions may have shifted from erasure)
        mfmaRanges.clear();
        {
          // Find s_setprio 0 again
          size_t sp0New = loop.find("s_setprio 0", sp0search);
          if (sp0New == std::string::npos) {
            sp0search = sp0BlockEnd;
            continue;
          }
          size_t sp0NewLineStart = loop.rfind('\n', sp0New);
          sp0NewLineStart = (sp0NewLineStart == std::string::npos) ? 0 : sp0NewLineStart + 1;

          size_t asmCheck = loop.rfind(";;#ASMSTART", sp0New);
          size_t mfmaEnd = sp0NewLineStart;
          if (asmCheck != std::string::npos && sp0New - asmCheck < 30) {
            size_t asmLine = loop.rfind('\n', asmCheck);
            if (asmLine != std::string::npos) mfmaEnd = asmLine + 1;
          }

          size_t cur = mfmaEnd;
          while (mfmaRanges.size() < 16 && cur > 0) {
            cur--;
            size_t lineStart = loop.rfind('\n', cur);
            lineStart = (lineStart == std::string::npos) ? 0 : lineStart + 1;
            std::string line = loop.substr(lineStart, cur + 1 - lineStart);
            std::string trimmed = line;
            while (!trimmed.empty() && (trimmed[0] == ' ' || trimmed[0] == '\t'))
              trimmed.erase(0, 1);
            if (trimmed.find("v_mfma") == 0) {
              size_t eol = loop.find('\n', lineStart);
              eol = (eol == std::string::npos) ? loop.size() : eol + 1;
              mfmaRanges.push_back({lineStart, eol});
              cur = lineStart;
              if (lineStart > 0) cur--;
            } else if (trimmed.empty()) {
              cur = lineStart;
              if (lineStart > 0) cur--;
            } else {
              break;
            }
          }
          std::reverse(mfmaRanges.begin(), mfmaRanges.end());
        }

        if (mfmaRanges.size() < 16) {
          sp0search = sp0BlockEnd;
          continue;
        }

        // Insert V reads after specified MFMA positions (insert in reverse to preserve offsets)
        for (int i = 3; i >= 0; i--) {
          int afterMfma = insertPoints[i] - 1; // 0-indexed
          if (afterMfma >= (int)mfmaRanges.size()) continue;
          size_t insertPos = mfmaRanges[afterMfma].second;
          std::string ins = vreadLines[i] + "\n";
          loop.insert(insertPos, ins);
          // Adjust later MFMA ranges
          for (int j = afterMfma + 1; j < (int)mfmaRanges.size(); j++) {
            mfmaRanges[j].first += ins.size();
            mfmaRanges[j].second += ins.size();
          }
        }

        vreadsMoved += 4;
        llvm::errs() << "[postProcessISA] Pass 25: Moved 4 V reads into"
                     << " GEMM1 MFMA chain in " << label << "\n";

        // Find new s_setprio 0 position for next iteration
        size_t newSp0 = loop.find("s_setprio 0", mfmaRanges.back().second);
        sp0search = (newSp0 != std::string::npos) ? newSp0 + 20 : loop.size();
      }

      if (vreadsMoved > 0) {
        llvm::errs() << "[postProcessISA] Pass 25: " << vreadsMoved
                     << " V reads moved total in " << label << "\n";
      }
    }

    result = before + loop + after;
  }

  // Update VGPR metadata if restructureGEMM2_V2 allocated higher registers
  {
    auto updateVgprMeta = [](std::string &s, const std::string &tag, int newVal) {
      size_t pos = s.find(tag);
      while (pos != std::string::npos) {
        size_t numStart = pos + tag.size();
        while (numStart < s.size() && s[numStart] == ' ') numStart++;
        size_t numEnd = numStart;
        while (numEnd < s.size() && isdigit(s[numEnd])) numEnd++;
        if (numEnd > numStart) {
          int cur = atoi(s.c_str() + numStart);
          if (newVal > cur) {
            std::string nv = std::to_string(newVal);
            s.replace(numStart, numEnd - numStart, nv);
          }
        }
        pos = s.find(tag, numStart + 1);
      }
    };
    // The static g_maxNewVgpr is accessible here since we're in the same function
    // We need to read from the static declared in the loop body above.
    // Since C++ doesn't allow extern on a local static, just re-search the ISA
    // for the current vgpr count and conditionally update.
    // Actually, we can just look for any v[235+] references to determine if update needed.
    bool needsVgprUpdate = false;
    int maxV = 0;
    for (int v = 253; v >= 235; v--) {
      if (result.find("v[" + std::to_string(v) + ":") != std::string::npos ||
          result.find("v" + std::to_string(v) + ",") != std::string::npos ||
          result.find("v" + std::to_string(v) + " ") != std::string::npos) {
        maxV = v + 1;
        needsVgprUpdate = true;
        break;
      }
    }
    if (needsVgprUpdate && maxV > 0) {
      updateVgprMeta(result, ".amdhsa_next_free_vgpr", maxV);
      updateVgprMeta(result, ".vgpr_count:", maxV);
    }
  }

  // --- Pass 15: Interleave loop-top exp burst with K reads ---
  // Handles two kernel shapes:
  //   (A) 16x v_exp_f32 followed by 8x ds_read_b128
  //   (B) 8x ds_read_b128 followed by an optional s_setprio 1 block,
  //       a short prefix (for example, v_mul_f32), then 16x v_exp_f32
  // Reorder both forms to: 1 K_read + 2 exp repeated 8 times.
  // This spaces consecutive ds_read_b128 by ~12 cycles instead of issuing
  // all 8 reads back-to-back at the loop top.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };
    auto isExpLine = [&](const std::string &line) -> bool {
      return trim(line).find("v_exp_f32") == 0;
    };
    auto isDsReadB128Line = [&](const std::string &line) -> bool {
      return trim(line).find("ds_read_b128") == 0;
    };
    auto isSetprio1Line = [&](const std::string &line) -> bool {
      return trim(line) == "s_setprio 1";
    };
    auto isAsmStartLine = [&](const std::string &line) -> bool {
      return trim(line) == ";;#ASMSTART";
    };
    auto isAsmEndLine = [&](const std::string &line) -> bool {
      return trim(line) == ";;#ASMEND";
    };
    auto isLabelLine = [&](const std::string &line) -> bool {
      auto t = trim(line);
      return !t.empty() && t.back() == ':';
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 23 < allLines.size(); ) {
      size_t advance = 0;
      auto rewriteExpThenRead = [&](size_t start) -> bool {
        bool hasExp16 = true;
        for (size_t e = 0; e < 16; e++) {
          if (!isExpLine(allLines[start + e])) {
            hasExp16 = false;
            break;
          }
        }
        if (!hasExp16)
          return false;

        bool hasKRead8 = true;
        for (size_t k = 0; k < 8; k++) {
          if (!isDsReadB128Line(allLines[start + 16 + k])) {
            hasKRead8 = false;
            break;
          }
        }
        if (!hasKRead8)
          return false;

        std::vector<std::string> expLines(allLines.begin() + start,
                                          allLines.begin() + start + 16);
        std::vector<std::string> kLines(allLines.begin() + start + 16,
                                        allLines.begin() + start + 24);

        std::vector<std::string> interleaved;
        for (int g = 0; g < 8; g++) {
          interleaved.push_back(kLines[g]);
          interleaved.push_back(expLines[2 * g]);
          interleaved.push_back(expLines[2 * g + 1]);
        }

        for (int j = 0; j < 24; j++)
          allLines[start + j] = interleaved[j];

        llvm::errs() << "[postProcessISA] Pass 15: Interleaved exp-then-read "
                     << "1K+2exp x8 at line " << (start + 1) << "\n";
        modified = true;
        advance = 24;
        return true;
      };

      auto rewriteReadThenExp = [&](size_t start) -> bool {
        bool hasKRead8 = true;
        for (size_t k = 0; k < 8; k++) {
          if (!isDsReadB128Line(allLines[start + k])) {
            hasKRead8 = false;
            break;
          }
        }
        if (!hasKRead8)
          return false;

        size_t cursor = start + 8;
        std::vector<std::string> setprioBlock;
        if (cursor + 2 < allLines.size() && isAsmStartLine(allLines[cursor]) &&
            isSetprio1Line(allLines[cursor + 1]) &&
            isAsmEndLine(allLines[cursor + 2])) {
          setprioBlock.push_back(allLines[cursor]);
          setprioBlock.push_back(allLines[cursor + 1]);
          setprioBlock.push_back(allLines[cursor + 2]);
          cursor += 3;
        } else if (cursor < allLines.size() && isSetprio1Line(allLines[cursor])) {
          setprioBlock.push_back(allLines[cursor]);
          cursor++;
        }

        std::vector<std::string> prefixLines;
        while (cursor < allLines.size() && prefixLines.size() < 4 &&
               !isExpLine(allLines[cursor])) {
          auto t = trim(allLines[cursor]);
          if (t.empty()) {
            prefixLines.push_back(allLines[cursor]);
            cursor++;
            continue;
          }

          if (isDsReadB128Line(allLines[cursor]) || isSetprio1Line(allLines[cursor]) ||
              isAsmStartLine(allLines[cursor]) || isAsmEndLine(allLines[cursor]) ||
              isLabelLine(allLines[cursor]))
            return false;

          prefixLines.push_back(allLines[cursor]);
          cursor++;
        }

        if (cursor + 15 >= allLines.size())
          return false;
        for (size_t e = 0; e < 16; e++) {
          if (!isExpLine(allLines[cursor + e]))
            return false;
        }

        std::vector<std::string> kLines(allLines.begin() + start,
                                        allLines.begin() + start + 8);
        std::vector<std::string> expLines(allLines.begin() + cursor,
                                          allLines.begin() + cursor + 16);
        std::vector<std::string> rebuilt;
        if (!setprioBlock.empty() && prefixLines.empty()) {
          // The slot-1 exp collision stays in the late groups across all hot
          // prefix-free bursts, so keep the original split until group 6 has
          // cleared and only then re-raise priority for the final group.
          for (int g = 0; g < 7; g++) {
            rebuilt.push_back(kLines[g]);
            rebuilt.push_back(expLines[2 * g]);
            rebuilt.push_back(expLines[2 * g + 1]);
          }
          rebuilt.insert(rebuilt.end(), setprioBlock.begin(), setprioBlock.end());
          rebuilt.push_back(kLines[7]);
          rebuilt.push_back(expLines[14]);
          rebuilt.push_back(expLines[15]);
        } else {
          rebuilt.insert(rebuilt.end(), setprioBlock.begin(), setprioBlock.end());
          rebuilt.insert(rebuilt.end(), prefixLines.begin(), prefixLines.end());
          for (int g = 0; g < 8; g++) {
            rebuilt.push_back(kLines[g]);
            rebuilt.push_back(expLines[2 * g]);
            rebuilt.push_back(expLines[2 * g + 1]);
          }
        }

        size_t regionLen = cursor + 16 - start;
        if (rebuilt.size() != regionLen)
          return false;

        for (size_t j = 0; j < regionLen; j++)
          allLines[start + j] = rebuilt[j];

        llvm::errs() << "[postProcessISA] Pass 15: Interleaved read-then-exp "
                     << "1K+2exp x8 at line " << (start + 1)
                     << " (prefix=" << prefixLines.size()
                     << ", setprio=" << setprioBlock.size() << ")\n";
        modified = true;
        advance = regionLen;
        return true;
      };

      if (rewriteReadThenExp(i)) {
        i += advance;
        continue;
      }
      if (rewriteExpThenRead(i)) {
        i += advance;
        continue;
      }
      i++;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size()) result += "\n";
      }
    }
  }

  // --- Pass 15b: Pull the first ds_write ahead of vmcnt(0) in the tail ---
  // Transition-tail shape in the current best loop:
  //   s_barrier
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 v122 ...
  //   v_perm_b32 v123 ...
  //   ds_write_b64 v132, v[120:121]
  //   ds_write_b64 v133, v[122:123]
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  // The first ds_write only depends on v[120:121], which are already produced
  // before the preceding barrier. Issue it immediately after that barrier so the
  // LDS write can overlap with the later vmcnt(0) wait and the second permute pair.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 1; i + 6 < allLines.size(); i++) {
      if (trim(allLines[i - 1]) != "s_barrier")
        continue;
      if (trim(allLines[i]) != "s_waitcnt vmcnt(0)")
        continue;
      if (trim(allLines[i + 1]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 2]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 3]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 4]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 5]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 6]) != "s_barrier")
        continue;

      std::string firstWrite = allLines[i + 3];
      allLines.erase(allLines.begin() + i + 3);
      allLines.insert(allLines.begin() + i, firstWrite);
      llvm::errs() << "[postProcessISA] Pass 15b: Pulled first ds_write before "
                   << "vmcnt(0) at line " << (i + 1) << "\n";
      modified = true;
      i += 6;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15c: Pull the first Body-B ds_write ahead of the pre-store barrier ---
  // Hot-path shape after the 112T V-buffer split:
  //   s_waitcnt vmcnt(6)
  //   v_perm_b32 v120 ...
  //   s_waitcnt vmcnt(4)
  //   v_perm_b32 v121 ...
  //   s_barrier
  //   ds_write_b64 v131, v[120:121] offset:26112
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 v122 ...
  //   v_perm_b32 v123 ...
  //   ds_write_b64 v131, v[122:123] offset:30272
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  // The first ds_write only depends on the first permute pair and writes into the
  // alternate V LDS buffer. Issue it before the barrier so part of V staging lands
  // before the costly cross-wave sync without changing the later visibility barrier.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 11 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(6)")
        continue;
      if (trim(allLines[i + 1]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 2]) != "s_waitcnt vmcnt(4)" ||
          trim(allLines[i + 3]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 4]) != "s_barrier" ||
          trim(allLines[i + 5]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 6]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 7]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 8]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 9]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 10]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 11]) != "s_barrier")
        continue;

      std::string firstWrite = allLines[i + 5];
      allLines.erase(allLines.begin() + i + 5);
      allLines.insert(allLines.begin() + i + 4, firstWrite);
      llvm::errs() << "[postProcessISA] Pass 15c: Pulled first ds_write before "
                   << "pre-store barrier at line " << (i + 1) << "\n";
      modified = true;
      i += 11;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15d: Collapse the mid-tail V-staging barrier when the second
  // permute pair reuses the same VMEM sources as the first pair.
  // Current hot-path shape in the tail:
  //   s_waitcnt vmcnt(6)
  //   v_perm_b32 A0, srcX, srcY, s80/s81
  //   s_waitcnt vmcnt(4)
  //   v_perm_b32 A1, srcP, srcQ, s80/s81
  //   ds_write_b64 ..., A0/A1
  //   s_barrier
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 B0, srcX, srcY, s81/s80
  //   v_perm_b32 B1, srcP, srcQ, s81/s80
  //   ds_write_b64 ..., B0/B1
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  // The second permute pair only changes the selector SGPR and reuses the same
  // VMEM source VGPRs, so it is already safe once vmcnt(4) has been reached.
  // Move that second pair above the mid-tail barrier and keep the final barrier
  // as the only LDS-visibility sync point.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto splitComma = [&](const std::string &line) {
      std::vector<std::string> parts;
      std::string cur;
      for (char c : trim(line)) {
        if (c == ',') {
          parts.push_back(cur);
          cur.clear();
        } else {
          cur.push_back(c);
        }
      }
      parts.push_back(cur);
      for (auto &p : parts) {
        size_t s = 0;
        while (s < p.size() && (p[s] == ' ' || p[s] == '\t'))
          s++;
        size_t e = p.size();
        while (e > s && (p[e - 1] == ' ' || p[e - 1] == '\t'))
          e--;
        p = p.substr(s, e - s);
      }
      return parts;
    };

    auto samePermSources = [&](const std::string &a, const std::string &b) {
      auto pa = splitComma(a);
      auto pb = splitComma(b);
      if (pa.size() != 4 || pb.size() != 4)
        return false;
      if (pa[0].find("v_perm_b32") != 0 || pb[0].find("v_perm_b32") != 0)
        return false;
      return pa[1] == pb[1] && pa[2] == pb[2];
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 11 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(6)")
        continue;
      if (trim(allLines[i + 1]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 2]) != "s_waitcnt vmcnt(4)" ||
          trim(allLines[i + 3]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 4]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 5]) != "s_barrier" ||
          trim(allLines[i + 6]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 7]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 8]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 9]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 10]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 11]) != "s_barrier")
        continue;
      if (!samePermSources(allLines[i + 1], allLines[i + 7]) ||
          !samePermSources(allLines[i + 3], allLines[i + 8]))
        continue;

      std::vector<std::string> repl = {
          allLines[i],     allLines[i + 1], allLines[i + 2],  allLines[i + 3],
          allLines[i + 4], allLines[i + 7], allLines[i + 8],  allLines[i + 9],
          allLines[i + 6], allLines[i + 10], allLines[i + 11],
      };
      allLines.erase(allLines.begin() + i, allLines.begin() + i + 12);
      allLines.insert(allLines.begin() + i, repl.begin(), repl.end());

      llvm::errs() << "[postProcessISA] Pass 15d: Removed mid-tail barrier in "
                   << "V staging at line " << (i + 1) << "\n";
      modified = true;
      i += repl.size() - 1;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15e: Move the remaining lgkmcnt(0) wait behind the final V-staging
  // barrier in the collapsed-tail shape. This lets waves rendezvous at the
  // barrier earlier while preserving the per-wave LDS completion check before
  // the subsequent ds_read burst.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 10 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(6)")
        continue;
      if (trim(allLines[i + 1]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 2]) != "s_waitcnt vmcnt(4)" ||
          trim(allLines[i + 3]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 4]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 5]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 6]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 7]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 8]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 9]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 10]) != "s_barrier")
        continue;

      std::swap(allLines[i + 9], allLines[i + 10]);
      llvm::errs() << "[postProcessISA] Pass 15e: Moved lgkmcnt wait behind "
                   << "final V-staging barrier at line " << (i + 1) << "\n";
      modified = true;
      i += 10;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15e2: Remove the pre-store barrier when both tail permute pairs are
  // already materialized before it.
  // Current .LBB0_19 hot-path shape:
  //   s_waitcnt vmcnt(6)
  //   v_perm_b32 A0 ...
  //   v_perm_b32 A1 ...
  //   s_waitcnt vmcnt(4)
  //   v_perm_b32 B0 ...
  //   v_perm_b32 B1 ...
  //   s_barrier
  //   s_waitcnt vmcnt(0)
  //   ds_write_b64 ..., A0/A1
  //   ds_write_b64 ..., B0/B1
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ds_read_b128 ...
  // Both permute pairs are already complete before the first barrier, and the
  // final barrier still provides the only LDS-visibility rendezvous for the
  // following ds_read burst. Drop the pre-store barrier so waves can continue
  // into the LDS write handoff immediately.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 12 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(6)")
        continue;
      if (!startsWith(allLines[i + 1], "v_perm_b32") ||
          !startsWith(allLines[i + 2], "v_perm_b32") ||
          trim(allLines[i + 3]) != "s_waitcnt vmcnt(4)" ||
          !startsWith(allLines[i + 4], "v_perm_b32") ||
          !startsWith(allLines[i + 5], "v_perm_b32") ||
          trim(allLines[i + 6]) != "s_barrier" ||
          trim(allLines[i + 7]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 8], "ds_write_b64") ||
          !startsWith(allLines[i + 9], "ds_write_b64") ||
          trim(allLines[i + 10]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 11]) != "s_barrier" ||
          !startsWith(allLines[i + 12], "ds_read_b128"))
        continue;

      allLines.erase(allLines.begin() + i + 6);
      llvm::errs() << "[postProcessISA] Pass 15e2: Removed pre-store barrier "
                   << "before tail ds_write handoff at line " << (i + 1)
                   << "\n";
      modified = true;
      i += 11;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15e3: Move the trailing lgkmcnt(0) behind the split-loop handoff
  // barrier once both LDS write pairs have already been issued.
  // This split-loop handoff remains the 122T large-sequence baseline hot path when
  // the frontend takes prefix/tail split by default.
  // Current hot-path shape:
  //   s_waitcnt vmcnt(6)
  //   v_perm_b32 A0 ...
  //   v_perm_b32 B0 ...
  //   s_waitcnt vmcnt(4)
  //   v_perm_b32 A1 ...
  //   v_perm_b32 B1 ...
  //   s_waitcnt vmcnt(0)
  //   ds_write_b64 ..., A0/A1
  //   ds_write_b64 ..., B0/B1
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ds_read_b128 ...
  // The barrier is the cross-wave rendezvous point; the lgkmcnt wait only guards
  // the following LDS read burst. Swapping them lets waves reach the barrier
  // sooner without changing the eventual per-wave LDS readiness check.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 11 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(6)")
        continue;
      if (!startsWith(allLines[i + 1], "v_perm_b32") ||
          !startsWith(allLines[i + 2], "v_perm_b32") ||
          trim(allLines[i + 3]) != "s_waitcnt vmcnt(4)" ||
          !startsWith(allLines[i + 4], "v_perm_b32") ||
          !startsWith(allLines[i + 5], "v_perm_b32") ||
          trim(allLines[i + 6]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 7], "ds_write_b64") ||
          !startsWith(allLines[i + 8], "ds_write_b64") ||
          trim(allLines[i + 9]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 10]) != "s_barrier" ||
          !startsWith(allLines[i + 11], "ds_read_b128"))
        continue;

      std::swap(allLines[i + 9], allLines[i + 10]);
      llvm::errs() << "[postProcessISA] Pass 15e3: Moved split-loop lgkmcnt "
                   << "behind barrier at line " << (i + 1) << "\n";
      modified = true;
      i += 10;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15f: Pull the first pre-exp LDS write ahead of the final vmcnt(0)
  // in the prefixed pre-18852 handoff.
  // Hot-path shape:
  //   s_waitcnt vmcnt(2)
  //   v_perm_b32 A0 ...
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 A1 ...
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 B0 ...
  //   v_perm_b32 B1 ...
  //   ds_write_b64 ..., A0/A1
  //   ds_write_b64 ..., B0/B1
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ;;#ASMSTART / s_setprio 1 / ;;#ASMEND
  // The first ds_write only depends on the completed A0/A1 permute pair. Issue
  // it before the second vmcnt(0) and the later B0/B1 permutes so the LDS write
  // can overlap the remaining VMEM drain and permute work instead of landing
  // immediately in front of the barrier.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 13 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(2)")
        continue;
      if (trim(allLines[i + 1]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 2]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 3]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 4]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 5]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 6]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 7]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 8]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 9]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 10]) != "s_barrier" ||
          trim(allLines[i + 11]) != ";;#ASMSTART" ||
          trim(allLines[i + 12]) != "s_setprio 1" ||
          trim(allLines[i + 13]) != ";;#ASMEND")
        continue;

      std::string firstWrite = allLines[i + 7];
      allLines.erase(allLines.begin() + i + 7);
      allLines.insert(allLines.begin() + i + 4, firstWrite);
      llvm::errs() << "[postProcessISA] Pass 15f: Pulled first pre-exp ds_write "
                   << "before trailing vmcnt(0) at line " << (i + 1) << "\n";
      modified = true;
      i += 13;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15g: Start the first pre-20936 V LDS write inside the trailing
  // pk_mul chain instead of after the whole chain drains.
  // Current hot-path shape:
  //   v_pk_mul x32
  //   s_waitcnt vmcnt(6)
  //   v_perm_b32 ...
  //   s_waitcnt vmcnt(4)
  //   v_perm_b32 ...
  //   ds_write_b64 ...
  //   v_perm_b32 ...
  //   v_perm_b32 ...
  //   ds_write_b64 ...
  //   s_waitcnt vmcnt(0)
  //   s_barrier
  // The first permute pair and ds_write only serve the next V staging handoff.
  // Insert them ahead of the final eight pk_mul ops so the LDS write can
  // overlap the remaining pk_mul tail.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto isPkMul = [&](const std::string &line) -> bool {
      return trim(line).find("v_pk_mul_f32") == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 42 < allLines.size(); i++) {
      bool hasPkMul32 = true;
      for (size_t j = 0; j < 32; j++) {
        if (!isPkMul(allLines[i + j])) {
          hasPkMul32 = false;
          break;
        }
      }
      if (!hasPkMul32)
        continue;
      if (trim(allLines[i + 32]) != "s_waitcnt vmcnt(6)" ||
          trim(allLines[i + 33]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 34]) != "s_waitcnt vmcnt(4)" ||
          trim(allLines[i + 35]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 36]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 37]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 38]).find("v_perm_b32") != 0 ||
          trim(allLines[i + 39]).find("ds_write_b64") != 0 ||
          trim(allLines[i + 40]) != "s_waitcnt vmcnt(0)" ||
          trim(allLines[i + 41]) != "s_barrier")
        continue;

      std::vector<std::string> moved = {
          allLines[i + 32], allLines[i + 33], allLines[i + 34],
          allLines[i + 35], allLines[i + 36],
      };
      allLines.erase(allLines.begin() + i + 32, allLines.begin() + i + 37);
      allLines.insert(allLines.begin() + i + 24, moved.begin(), moved.end());
      llvm::errs() << "[postProcessISA] Pass 15g: Pulled first pre-barrier "
                   << "V staging block into pk_mul tail at line " << (i + 1)
                   << "\n";
      modified = true;
      i += 42;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15h: Pull the first pre-18852 LDS write ahead of the second
  // vmcnt(0) in the common-path handoff.
  // Current hot-path shape:
  //   s_waitcnt vmcnt(2)
  //   v_perm_b32 v120, ...
  //   v_perm_b32 v144, ...
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 v121, ...
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 v145, ...
  //   ds_write_b64 ..., v[120:121]
  //   ds_write_b64 ..., v[144:145]
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  // The first ds_write only depends on v120/v121, which are already finalized
  // after the third permute. Issue it before the second vmcnt(0) so the LDS
  // write overlaps the remaining VMEM drain instead of landing immediately in
  // front of the barrier.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 10 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_waitcnt vmcnt(2)")
        continue;
      if (!startsWith(allLines[i + 1], "v_perm_b32") ||
          !startsWith(allLines[i + 2], "v_perm_b32") ||
          trim(allLines[i + 3]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 4], "v_perm_b32") ||
          trim(allLines[i + 5]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 6], "v_perm_b32") ||
          trim(allLines[i + 7]).find("ds_write_b64 v131, v[120:121] offset:17408") != 0 ||
          trim(allLines[i + 8]).find("ds_write_b64 v131, v[144:145] offset:21568") != 0 ||
          trim(allLines[i + 9]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 10]) != "s_barrier")
        continue;

      std::string firstWrite = allLines[i + 7];
      allLines.erase(allLines.begin() + i + 7);
      allLines.insert(allLines.begin() + i + 5, firstWrite);
      llvm::errs() << "[postProcessISA] Pass 15h: Pulled first common-path "
                   << "pre-18852 ds_write ahead of trailing vmcnt(0) at line "
                   << (i + 1) << "\n";
      modified = true;
      i += 10;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15i: Move the pre-18852 lgkmcnt(0) wait behind the common-path
  // handoff barrier so waves can rendezvous earlier.
  // Current hot-path shape after Pass 15h:
  //   ... ds_write_b64 v[120:121]
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 v145, ...
  //   ds_write_b64 v[144:145]
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ds_read_b128 v[142:145], v132
  // The barrier is the cross-wave rendezvous point, while the lgkmcnt wait only
  // guards the following per-wave ds_read burst. Swap them so waves reach the
  // barrier sooner, then pay the remaining LDS completion cost after barrier.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 6 < allLines.size(); i++) {
      if (trim(allLines[i]).find("ds_write_b64 v131, v[120:121] offset:17408") != 0 ||
          trim(allLines[i + 1]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 2], "v_perm_b32") ||
          trim(allLines[i + 3]).find("ds_write_b64 v131, v[144:145] offset:21568") != 0 ||
          trim(allLines[i + 4]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 5]) != "s_barrier" ||
          !startsWith(allLines[i + 6], "ds_read_b128 v[142:145], v132"))
        continue;

      std::swap(allLines[i + 4], allLines[i + 5]);
      llvm::errs() << "[postProcessISA] Pass 15i: Moved common-path pre-18852 "
                   << "lgkmcnt wait behind barrier at line " << (i + 1)
                   << "\n";
      modified = true;
      i += 6;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15j: Delay the prefixed pre-exp lgkmcnt(0) until just before the
  // first ds_read burst.
  // Current hot-path shape:
  //   ds_write_b64 ..., v[196:197]
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ;;#ASMSTART
  //   s_setprio 1
  //   ;;#ASMEND
  //   v_mul_f32_e32 ...
  //   ds_read_b128 v[196:199], v132
  // The lgkm wait only guards the upcoming ds_read burst, while the barrier,
  // setprio handoff, and prefix v_mul are independent. Move the wait down so
  // those common-path instructions execute before paying the LDS completion cost.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 7 < allLines.size(); i++) {
      if (trim(allLines[i]).find("ds_write_b64 v131, v[196:197] offset:21568") != 0 ||
          trim(allLines[i + 1]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 2]) != "s_barrier" ||
          trim(allLines[i + 3]) != ";;#ASMSTART" ||
          trim(allLines[i + 4]) != "s_setprio 1" ||
          trim(allLines[i + 5]) != ";;#ASMEND" ||
          !startsWith(allLines[i + 6], "v_mul_f32_e32") ||
          !startsWith(allLines[i + 7], "ds_read_b128 v[196:199], v132"))
        continue;

      std::string wait = allLines[i + 1];
      allLines.erase(allLines.begin() + i + 1);
      allLines.insert(allLines.begin() + i + 6, wait);
      llvm::errs() << "[postProcessISA] Pass 15j: Delayed prefixed pre-exp "
                   << "lgkmcnt wait to just before ds_read at line "
                   << (i + 1) << "\n";
      modified = true;
      i += 7;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15j2: Delay the pre-exp lgkmcnt(0) until just before the first
  // ds_read burst in the common split-loop handoff.
  // Current hot-path shape:
  //   ds_write_b64 ...
  //   ds_write_b64 ...
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ;;#ASMSTART
  //   s_setprio 1
  //   ;;#ASMEND
  //   v_mul_f32_e32 ...
  //   ds_read_b128 ...
  // The barrier is the cross-wave rendezvous. The lgkm wait only guards the
  // following per-wave ds_read, so move it below the barrier handoff and the
  // common v_mul setup.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 8 < allLines.size(); i++) {
      if (!startsWith(allLines[i], "ds_write_b64") ||
          !startsWith(allLines[i + 1], "ds_write_b64") ||
          trim(allLines[i + 2]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 3]) != "s_barrier" ||
          trim(allLines[i + 4]) != ";;#ASMSTART" ||
          trim(allLines[i + 5]) != "s_setprio 1" ||
          trim(allLines[i + 6]) != ";;#ASMEND" ||
          !startsWith(allLines[i + 7], "v_mul_f32_e32") ||
          !startsWith(allLines[i + 8], "ds_read_b128"))
        continue;

      std::string wait = allLines[i + 2];
      allLines.erase(allLines.begin() + i + 2);
      allLines.insert(allLines.begin() + i + 7, wait);
      llvm::errs() << "[postProcessISA] Pass 15j2: Delayed split-loop pre-exp "
                   << "lgkmcnt wait behind barrier at line " << (i + 1)
                   << "\n";
      modified = true;
      i += 8;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15j3: Delay the staggered pre-exp lgkmcnt(0) until just before the
  // first ds_read burst when the second ds_write lands after a vmcnt(0)+perm
  // pair.
  // Current hot-path shape:
  //   ds_write_b64 ...
  //   s_waitcnt vmcnt(0)
  //   v_perm_b32 ...
  //   v_perm_b32 ...
  //   ds_write_b64 ...
  //   s_waitcnt lgkmcnt(0)
  //   s_barrier
  //   ;;#ASMSTART
  //   s_setprio 1
  //   ;;#ASMEND
  //   v_mul_f32_e32 ...
  //   ds_read_b128 ...
  // Both ds_write instructions have already issued before the barrier. The
  // remaining lgkm wait only guards the following per-wave ds_read burst, so
  // move it below the barrier handoff and common v_mul setup.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    auto startsWith = [&](const std::string &line, const char *prefix) -> bool {
      return trim(line).find(prefix) == 0;
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 11 < allLines.size(); i++) {
      if (!startsWith(allLines[i], "ds_write_b64") ||
          trim(allLines[i + 1]) != "s_waitcnt vmcnt(0)" ||
          !startsWith(allLines[i + 2], "v_perm_b32") ||
          !startsWith(allLines[i + 3], "v_perm_b32") ||
          !startsWith(allLines[i + 4], "ds_write_b64") ||
          trim(allLines[i + 5]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 6]) != "s_barrier" ||
          trim(allLines[i + 7]) != ";;#ASMSTART" ||
          trim(allLines[i + 8]) != "s_setprio 1" ||
          trim(allLines[i + 9]) != ";;#ASMEND" ||
          !startsWith(allLines[i + 10], "v_mul_f32_e32") ||
          !startsWith(allLines[i + 11], "ds_read_b128"))
        continue;

      std::string wait = allLines[i + 5];
      allLines.erase(allLines.begin() + i + 5);
      allLines.insert(allLines.begin() + i + 10, wait);
      llvm::errs() << "[postProcessISA] Pass 15j3: Delayed staggered pre-exp "
                   << "lgkmcnt wait behind barrier at line " << (i + 1)
                   << "\n";
      modified = true;
      i += 11;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15k: Fill the post-20936 GEMM2 lgkm wait with common-path SALU.
  // Current hot-path shape:
  //   s_and_b64 s[0:1], vcc, exec
  //   s_waitcnt lgkmcnt(0)
  //   v_mfma_f32_32x32x8_bf16 v[48:63], v[118:119], v[176:177], v[48:63]
  //   ds_read_b64 ...
  //   ds_read_b64 ...
  //   ds_read_b64 ...
  //   ds_read_b64 ...
  //   s_cselect_b32 s0, s52, s35
  //   s_mov_b32 m0, s43
  //   s_mul_i32 s0, s63, s0
  // The SALU trio only prepares the later branch-local buffer_load soffset/m0
  // state. Pull it in front of the lgkm wait so the wait window does useful
  // work without perturbing the MFMA/VALU cadence.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 9 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_and_b64 s[0:1], vcc, exec" ||
          trim(allLines[i + 1]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 2]).find(
              "v_mfma_f32_32x32x8_bf16 v[48:63], v[118:119], v[176:177], "
              "v[48:63]") != 0 ||
          trim(allLines[i + 3]).find("ds_read_b64 v[118:119], v128 offset:26240") != 0 ||
          trim(allLines[i + 4]).find("ds_read_b64 v[150:151], v128 offset:27264") != 0 ||
          trim(allLines[i + 5]).find("ds_read_b64 v[152:153], v128 offset:28288") != 0 ||
          trim(allLines[i + 6]).find("ds_read_b64 v[154:155], v128 offset:29312") != 0 ||
          trim(allLines[i + 7]) != "s_cselect_b32 s0, s52, s35" ||
          trim(allLines[i + 8]) != "s_mov_b32 m0, s43" ||
          trim(allLines[i + 9]) != "s_mul_i32 s0, s63, s0")
        continue;

      std::vector<std::string> moved = {allLines[i + 7], allLines[i + 8],
                                        allLines[i + 9]};
      allLines.erase(allLines.begin() + i + 7, allLines.begin() + i + 10);
      allLines.insert(allLines.begin() + i + 1, moved.begin(), moved.end());
      llvm::errs() << "[postProcessISA] Pass 15k: Hoisted common-path SALU "
                   << "into post-20936 lgkm wait at line " << (i + 1) << "\n";
      modified = true;
      i += 9;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15l: Start the common-path V buffer-load quartet earlier.
  // Current hot-path shape in the post-18852 tail:
  //   v_mfma_f32_32x32x8_bf16 v[48:63], v[144:145], v[180:181], v[48:63]
  //   ds_read_b64 v[142:143], ...
  //   ds_read_b64 v[144:145], ...
  //   ds_read_b64 v[162:163], ...
  //   buffer_load_dword v164, v137, s[48:51], s55 offen
  //   buffer_load_dword v165, v138, s[48:51], s55 offen
  //   buffer_load_dword v166, v139, s[48:51], s55 offen
  //   buffer_load_dword v167, v140, s[48:51], s55 offen
  // Hoist the common-path VMEM quartet right after the first MFMA so it can
  // overlap a little earlier with the following MFMA/VALU work, while keeping
  // the branch-local LDS-backed loads in place.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 7 < allLines.size(); i++) {
      if (trim(allLines[i]).find(
              "v_mfma_f32_32x32x8_bf16 v[48:63], v[144:145], v[180:181], "
              "v[48:63]") != 0 ||
          trim(allLines[i + 1]).find("ds_read_b64 v[142:143], v128 offset:18816") != 0 ||
          trim(allLines[i + 2]).find("ds_read_b64 v[144:145], v128 offset:19840") != 0 ||
          trim(allLines[i + 3]).find("ds_read_b64 v[162:163], v128 offset:20864") != 0 ||
          trim(allLines[i + 4]).find("buffer_load_dword v164, v137, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 5]).find("buffer_load_dword v165, v138, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 6]).find("buffer_load_dword v166, v139, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 7]).find("buffer_load_dword v167, v140, s[48:51], s55 offen") != 0)
        continue;

      std::vector<std::string> moved = {allLines[i + 4], allLines[i + 5],
                                        allLines[i + 6], allLines[i + 7]};
      allLines.erase(allLines.begin() + i + 4, allLines.begin() + i + 8);
      allLines.insert(allLines.begin() + i + 1, moved.begin(), moved.end());
      llvm::errs() << "[postProcessISA] Pass 15l: Hoisted common-path V "
                   << "buffer-load quartet after first post-18852 MFMA at line "
                   << (i + 1) << "\n";
      modified = true;
      i += 7;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15m: Let the hot post-barrier MFMA issue before the second LDS-backed
  // V load pair finishes setting up. In both common-path handoff tails we have:
  //   buffer_load_dword v133, ... offen lds
  //   s_mov_b32 m0, ...
  //   [optional cndmask / s_nop]
  //   buffer_load_dword v134, ... offen lds
  //   s_mov_b32 m0, ...
  //   v_mfma_f32_32x32x8_bf16 v[32:47], ...
  // The MFMA is independent of the second LDS-backed load pair, while the load can
  // overlap with the MFMA. Swap that pair behind the MFMA so the MFMA starts sooner
  // without changing the earlier v133 issue or the later m0 state for v135.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 6 < allLines.size(); i++) {
      if (trim(allLines[i]).find("buffer_load_dword v133, s[44:47], s66 offen lds") == 0 &&
          trim(allLines[i + 1]) == "s_mov_b32 m0, s59" &&
          trim(allLines[i + 2]).find("v_cndmask_b32_e64 v74, v125, v74, s[20:21]") == 0 &&
          trim(allLines[i + 3]).find("buffer_load_dword v134, s[44:47], s66 offen lds") == 0 &&
          trim(allLines[i + 4]) == "s_mov_b32 m0, s61" &&
          trim(allLines[i + 5]).find(
              "v_mfma_f32_32x32x8_bf16 v[32:47], v[150:151], v[180:181], "
              "v[32:47]") == 0 &&
          trim(allLines[i + 6]).find("buffer_load_dword v135, s[44:47], s66 offen lds") == 0) {
        std::vector<std::string> moved = {allLines[i + 3], allLines[i + 4]};
        std::string mfma = allLines[i + 5];
        allLines.erase(allLines.begin() + i + 3, allLines.begin() + i + 6);
        allLines.insert(allLines.begin() + i + 3, mfma);
        allLines.insert(allLines.begin() + i + 4, moved.begin(), moved.end());
        llvm::errs() << "[postProcessISA] Pass 15m: Moved post-18852 second LDS "
                     << "load pair behind hot MFMA at line " << (i + 3) << "\n";
        modified = true;
        i += 6;
        continue;
      }

      if (trim(allLines[i]).find("buffer_load_dword v133, s[44:47], s0 offen lds") == 0 &&
          trim(allLines[i + 1]) == "s_mov_b32 m0, s65" &&
          trim(allLines[i + 2]) == "s_nop 0" &&
          trim(allLines[i + 3]).find("buffer_load_dword v134, s[44:47], s0 offen lds") == 0 &&
          trim(allLines[i + 4]) == "s_mov_b32 m0, s73" &&
          trim(allLines[i + 5]).find(
              "v_mfma_f32_32x32x8_bf16 v[32:47], v[150:151], v[178:179], "
              "v[32:47]") == 0 &&
          trim(allLines[i + 6]).find("buffer_load_dword v135, s[44:47], s0 offen lds") == 0) {
        std::vector<std::string> moved = {allLines[i + 3], allLines[i + 4]};
        std::string mfma = allLines[i + 5];
        allLines.erase(allLines.begin() + i + 3, allLines.begin() + i + 6);
        allLines.insert(allLines.begin() + i + 3, mfma);
        allLines.insert(allLines.begin() + i + 4, moved.begin(), moved.end());
        llvm::errs() << "[postProcessISA] Pass 15m: Moved post-20936 second LDS "
                     << "load pair behind hot MFMA at line " << (i + 3) << "\n";
        modified = true;
        i += 6;
      }
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15n: Fill the hot post-18852 MFMA->cndmask gap with common-path
  // VMEM loads.
  // Current hot-path shape:
  //   v_mfma_f32_32x32x8_bf16 v[16:31], v[142:143], v[178:179], v[16:31]
  //   ds_read_b64 v[142:143], v128 offset:17792
  //   v_cmp_lt_i32_e64 ...
  //   v_cmp_lt_i32_e64 ...
  //   v_cndmask_b32_e64 v68, ...
  //   v_cndmask_b32_e64 v69, ...
  //   ...
  //   v_mfma_f32_32x32x8_bf16 v[48:63], v[144:145], v[180:181], v[48:63]
  //   buffer_load_dword v164/v165/v166/v167, ...
  // The common-path VMEM quartet only feeds later tail work. Pull it up to right
  // after the first ds_read so those VMEM issues occupy the MFMA window before the
  // hottest slot0 cndmask pair.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 21 < allLines.size(); i++) {
      if (trim(allLines[i]).find(
              "v_mfma_f32_32x32x8_bf16 v[16:31], v[142:143], v[178:179], "
              "v[16:31]") != 0 ||
          trim(allLines[i + 1]).find("ds_read_b64 v[142:143], v128 offset:17792") != 0 ||
          trim(allLines[i + 2]).find("v_cmp_lt_i32_e64 s[12:13], 9, v127") != 0 ||
          trim(allLines[i + 3]).find("v_cmp_lt_i32_e64 s[14:15], 10, v127") != 0 ||
          trim(allLines[i + 4]).find("v_cndmask_b32_e64 v68, v125, v68, s[8:9]") != 0 ||
          trim(allLines[i + 5]).find("v_cndmask_b32_e64 v69, v125, v69, s[10:11]") != 0 ||
          trim(allLines[i + 6]).find("v_cmp_lt_i32_e64 s[16:17], 15, v127") != 0 ||
          trim(allLines[i + 7]).find("v_cmp_lt_i32_e64 s[18:19], 16, v127") != 0 ||
          trim(allLines[i + 8]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 9]).find(
              "v_mfma_f32_32x32x8_bf16 v[0:15], v[142:143], v[178:179], "
              "v[0:15]") != 0 ||
          trim(allLines[i + 17]).find(
              "v_mfma_f32_32x32x8_bf16 v[48:63], v[144:145], v[180:181], "
              "v[48:63]") != 0 ||
          trim(allLines[i + 18]).find("buffer_load_dword v164, v137, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 19]).find("buffer_load_dword v165, v138, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 20]).find("buffer_load_dword v166, v139, s[48:51], s55 offen") != 0 ||
          trim(allLines[i + 21]).find("buffer_load_dword v167, v140, s[48:51], s55 offen") != 0)
        continue;

      std::vector<std::string> moved = {
          allLines[i + 18], allLines[i + 19], allLines[i + 20], allLines[i + 21]};
      allLines.erase(allLines.begin() + i + 18, allLines.begin() + i + 22);
      allLines.insert(allLines.begin() + i + 2, moved.begin(), moved.end());
      llvm::errs() << "[postProcessISA] Pass 15n: Pulled post-18852 common-path "
                   << "VMEM quartet into MFMA->cndmask gap at line "
                   << (i + 1) << "\n";
      modified = true;
      i += 21;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 15o: Relax the split-prefix post-exp GEMM2 waits to only wait for
  // the first LDS-backed V pair that the next MFMA actually consumes.
  // Current hot-prefix shape in the final ISA:
  //   s_mov_b32 m0, s3
  //   s_waitcnt lgkmcnt(0)
  //   v_mfma ... v[48:63], v[148:149], ...
  //   ds_read_b64 v[148:149], ... offset:17536
  //   ds_read_b64 v[156:157], ... offset:18560
  //   ds_read_b64 v[158:159], ... offset:19584
  //   ds_read_b64 v[160:161], ... offset:20608
  //   s_waitcnt lgkmcnt(0)
  //   v_mfma ... v[32:47], v[148:149], ...
  //   ds_read_b64 v[148:149], ... offset:17664
  //   ds_read_b64 v[162:163], ... offset:18688
  //   ds_read_b64 v[164:165], ... offset:19712
  //   ds_read_b64 v[166:167], ... offset:20736
  //   s_waitcnt lgkmcnt(0)
  //   v_mfma ... v[16:31], v[148:149], ...
  // The next MFMA after each wait only consumes the first reloaded pair
  // v[148:149]. The later three ds_read_b64 ops feed later MFMAs, so keeping
  // three lgkm operations outstanding lets them overlap with the current MFMA
  // window instead of draining them all up front.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string line;
      while (std::getline(iss, line))
        allLines.push_back(line);
    }

    bool modified = false;
    for (size_t i = 0; i + 14 < allLines.size(); i++) {
      if (trim(allLines[i]) != "s_mov_b32 m0, s3" ||
          trim(allLines[i + 1]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 2]).find(
              "v_mfma_f32_32x32x8_bf16 v[48:63], v[148:149], v[184:185], "
              "v[48:63]") != 0 ||
          trim(allLines[i + 3]) != "ds_read_b64 v[148:149], v135 offset:17536" ||
          trim(allLines[i + 4]) != "ds_read_b64 v[156:157], v135 offset:18560" ||
          trim(allLines[i + 5]) != "ds_read_b64 v[158:159], v135 offset:19584" ||
          trim(allLines[i + 6]) != "ds_read_b64 v[160:161], v135 offset:20608" ||
          trim(allLines[i + 7]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 8]).find(
              "v_mfma_f32_32x32x8_bf16 v[32:47], v[148:149], v[184:185], "
              "v[32:47]") != 0 ||
          trim(allLines[i + 9]) != "ds_read_b64 v[148:149], v135 offset:17664" ||
          trim(allLines[i + 10]) != "ds_read_b64 v[162:163], v135 offset:18688" ||
          trim(allLines[i + 11]) != "ds_read_b64 v[164:165], v135 offset:19712" ||
          trim(allLines[i + 12]) != "ds_read_b64 v[166:167], v135 offset:20736" ||
          trim(allLines[i + 13]) != "s_waitcnt lgkmcnt(0)" ||
          trim(allLines[i + 14]).find(
              "v_mfma_f32_32x32x8_bf16 v[16:31], v[148:149], v[184:185], "
              "v[16:31]") != 0)
        continue;

      allLines[i + 1] = "\ts_waitcnt lgkmcnt(3)";
      allLines[i + 7] = "\ts_waitcnt lgkmcnt(3)";
      allLines[i + 13] = "\ts_waitcnt lgkmcnt(3)";
      llvm::errs() << "[postProcessISA] Pass 15o: Relaxed split-prefix GEMM2 "
                   << "wait trio to lgkmcnt(3) at line " << (i + 1) << "\n";
      modified = true;
      i += 14;
    }

    if (modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size())
          result += "\n";
      }
    }
  }

  // --- Pass 16: GEMM2 V-read defragmentation ---
  // DISABLED: Pre-loading all 16 V reads causes regression (114.7T → 109.9T).
  // Root cause: causal mask code between first and second MFMA still fragments
  // the chain, and the lost software-pipelining (interleaved V reads + MFMAs)
  // outweighs the reduced lgkmcnt stalls. Need to also move causal mask before
  // GEMM2 to match reference ASM structure for this to work.
  if (false) {
    static const int V_OFFSETS[] = {
      17408, 18432, 19456, 20480, 17536, 18560, 19584, 20608,
      17664, 18688, 19712, 20736, 17792, 18816, 19840, 20864
    };
    static const int NUM_V_OFFSETS = 16;

    std::vector<std::string> allLines;
    {
      std::istringstream iss(result);
      std::string l;
      while (std::getline(iss, l)) allLines.push_back(l);
    }

    auto trim = [](const std::string &s) -> std::string {
      size_t start = 0;
      while (start < s.size() && (s[start]==' '||s[start]=='\t')) start++;
      return s.substr(start);
    };

    auto isVRead = [&](const std::string &l) -> bool {
      auto t = trim(l);
      if (t.find("ds_read_b64") != 0) return false;
      for (int i = 0; i < NUM_V_OFFSETS; i++)
        if (t.find("offset:" + std::to_string(V_OFFSETS[i])) != std::string::npos)
          return true;
      return false;
    };

    auto parseRegPair = [](const std::string &s, size_t start)
        -> std::pair<int,int> {
      auto p = s.find("v[", start);
      if (p == std::string::npos) return {-1,-1};
      int lo = atoi(s.c_str() + p + 2);
      auto c = s.find(':', p + 2);
      if (c == std::string::npos) return {-1,-1};
      int hi = atoi(s.c_str() + c + 1);
      return {lo, hi};
    };

    auto parseOffset = [](const std::string &s) -> int {
      auto p = s.find("offset:");
      if (p == std::string::npos) return -1;
      return atoi(s.c_str() + p + 7);
    };

    bool pass16modified = false;

    llvm::errs() << "[Pass 16] allLines.size()=" << allLines.size() << "\n";

    for (size_t scan = 0; scan + 2 < allLines.size(); scan++) {
      if (trim(allLines[scan]) != "s_setprio 0") continue;
      llvm::errs() << "[Pass 16] Found s_setprio 0 at line " << scan+1
                   << ", next=" << trim(allLines[scan+1]).substr(0, 40) << "\n";
      if (trim(allLines[scan+1]).find(";;#ASMEND") != 0) continue;

      size_t regionStart = scan + 2;

      // Collect V reads in a window after setprio 0
      struct VR { size_t idx; int dLo, dHi, off; };
      std::vector<VR> vrs;
      for (size_t i = regionStart; i < std::min(regionStart+200, allLines.size()); i++) {
        if (isVRead(allLines[i])) {
          auto [lo,hi] = parseRegPair(allLines[i], 0);
          int off = parseOffset(allLines[i]);
          vrs.push_back({i, lo, hi, off});
        }
      }
      llvm::errs() << "[Pass 16] V reads found: " << vrs.size() << "/" << NUM_V_OFFSETS << "\n";
      if ((int)vrs.size() != NUM_V_OFFSETS) continue;

      // Skip if V reads are already contiguous (already defragmented)
      bool alreadyContiguous = true;
      for (int vi = 1; vi < (int)vrs.size(); vi++) {
        if (vrs[vi].idx != vrs[vi-1].idx + 1) {
          alreadyContiguous = false;
          break;
        }
      }
      if (alreadyContiguous) {
        llvm::errs() << "[Pass 16] V reads already contiguous, skipping\n";
        scan = vrs.back().idx;
        continue;
      }

      // Pre-compute: find consuming MFMA for each V read
      // The consuming MFMA is the first MFMA after the V read line
      // that uses v[dLo:dHi] as SrcA (2nd register argument).
      struct VRInfo {
        VR vr;
        size_t mfmaIdx;       // line index of consuming MFMA
        int newDLo, newDHi;   // remapped dest (-1 = keep original)
      };
      std::vector<VRInfo> vrInfo;

      for (auto &v : vrs) {
        auto srcA = "v[" + std::to_string(v.dLo) + ":" +
                    std::to_string(v.dHi) + "]";
        size_t found = 0;
        for (size_t j = v.idx + 1; j < std::min(v.idx + 300, allLines.size()); j++) {
          auto t = trim(allLines[j]);
          if (t.find("v_mfma_f32_32x32x8_bf16") != 0) continue;
          // Check SrcA: 2nd v[...] in the line
          auto p1 = t.find("v[");          // dest
          if (p1 == std::string::npos) continue;
          auto p2 = t.find("v[", p1 + 3);  // SrcA
          if (p2 == std::string::npos) continue;
          auto [sa_lo, sa_hi] = parseRegPair(t, p2);
          if (sa_lo == v.dLo && sa_hi == v.dHi) {
            found = j;
            break;
          }
        }
        vrInfo.push_back({v, found, -1, -1});
      }

      // Check all V reads found consuming MFMAs
      bool allOk = true;
      for (size_t vi_idx = 0; vi_idx < vrInfo.size(); vi_idx++) {
        auto &vi = vrInfo[vi_idx];
        if (vi.mfmaIdx == 0) {
          llvm::errs() << "[Pass 16] VR#" << vi_idx << " (v["
                       << vi.vr.dLo << ":" << vi.vr.dHi << "] off="
                       << vi.vr.off << " line=" << vi.vr.idx+1
                       << ") NO consuming MFMA found!\n";
          allOk = false; break;
        }
      }
      if (!allOk) {
        llvm::errs() << "[Pass 16] SKIP: not all V reads have MFMAs\n";
        continue;
      }

      // Build remap table for reused destination registers
      std::map<std::pair<int,int>, int> destCount;
      for (auto &vi : vrInfo) destCount[{vi.vr.dLo, vi.vr.dHi}]++;

      // Use VGPRs above the current max (238+) which are guaranteed free.
      // Must be even-aligned for ds_read_b64 (64-bit aligned VGPR pairs).
      // The kernel uses v[0:236], limit is 256 for occupancy=1.
      int nextFree = 238;
      std::map<std::pair<int,int>, bool> firstSeen;
      for (auto &vi : vrInfo) {
        auto key = std::make_pair(vi.vr.dLo, vi.vr.dHi);
        if (destCount[key] <= 1) continue;
        if (!firstSeen[key]) {
          firstSeen[key] = true;
        } else {
          vi.newDLo = nextFree;
          vi.newDHi = nextFree + 1;
          nextFree += 2;
        }
      }

      // Build new V read lines
      std::vector<std::string> newVReadLines;
      for (auto &vi : vrInfo) {
        std::string nl = allLines[vi.vr.idx];
        if (vi.newDLo >= 0) {
          auto oldD = "v[" + std::to_string(vi.vr.dLo) + ":" +
                      std::to_string(vi.vr.dHi) + "]";
          auto newD = "v[" + std::to_string(vi.newDLo) + ":" +
                      std::to_string(vi.newDHi) + "]";
          auto pos = nl.find(oldD);
          if (pos != std::string::npos) nl.replace(pos, oldD.size(), newD);
        }
        newVReadLines.push_back(nl);
      }

      // Update MFMA SrcA for remapped V reads
      for (auto &vi : vrInfo) {
        if (vi.newDLo < 0) continue;
        auto oldA = "v[" + std::to_string(vi.vr.dLo) + ":" +
                    std::to_string(vi.vr.dHi) + "]";
        auto newA = "v[" + std::to_string(vi.newDLo) + ":" +
                    std::to_string(vi.newDHi) + "]";
        auto &mline = allLines[vi.mfmaIdx];
        // Find 2nd v[ (SrcA position)
        auto p1 = mline.find("v[");
        if (p1 == std::string::npos) continue;
        auto p2 = mline.find("v[", p1 + 3);
        if (p2 == std::string::npos) continue;
        // Verify it matches
        if (mline.substr(p2, oldA.size()) == oldA)
          mline.replace(p2, oldA.size(), newA);
      }

      // Mark V read lines and V-read-related lgkmcnt for removal
      std::set<size_t> toRemove;
      for (auto &vi : vrInfo) toRemove.insert(vi.vr.idx);

      // Find lgkmcnt(3) lines between regionStart and last V read
      size_t lastVReadLine = vrs.back().idx;
      for (size_t i = regionStart; i <= lastVReadLine + 3 && i < allLines.size(); i++) {
        auto t = trim(allLines[i]);
        if (t.find("s_waitcnt lgkmcnt(3)") == 0) toRemove.insert(i);
        if (t.find("s_waitcnt lgkmcnt(0)") == 0) {
          // Only remove if before the first MFMA (V-read-related)
          if (i < vrInfo[0].mfmaIdx) toRemove.insert(i);
        }
      }

      // Find first and second GEMM2 MFMA lines
      // First MFMA only needs first V read → lgkmcnt(15)
      // Second MFMA (after causal mask) gets lgkmcnt(0) for full latency hiding
      size_t firstMfma = allLines.size();
      size_t secondMfma = allLines.size();
      for (auto &vi : vrInfo) {
        if (vi.mfmaIdx < firstMfma) {
          secondMfma = firstMfma;
          firstMfma = vi.mfmaIdx;
        } else if (vi.mfmaIdx < secondMfma && vi.mfmaIdx != firstMfma) {
          secondMfma = vi.mfmaIdx;
        }
      }

      // Rebuild: remove marked lines, insert V reads + progressive lgkmcnt
      std::vector<std::string> rebuilt;
      bool vReadsInserted = false;
      bool lgkm15Inserted = false;
      bool lgkm0Inserted = false;

      for (size_t i = 0; i < allLines.size(); i++) {
        if (i == regionStart && !vReadsInserted) {
          for (auto &vl : newVReadLines) rebuilt.push_back(vl);
          vReadsInserted = true;
        }
        if (toRemove.count(i)) continue;

        if (i == firstMfma && !lgkm15Inserted) {
          rebuilt.push_back("\ts_waitcnt lgkmcnt(15)");
          lgkm15Inserted = true;
        }
        if (i == secondMfma && !lgkm0Inserted) {
          rebuilt.push_back("\ts_waitcnt lgkmcnt(0)");
          lgkm0Inserted = true;
        }
        rebuilt.push_back(allLines[i]);
      }
      bool lgkmInserted = lgkm15Inserted; // for success check

      if (vReadsInserted && lgkmInserted) {
        allLines = std::move(rebuilt);
        pass16modified = true;
        int remapped = 0;
        for (auto &vi : vrInfo) if (vi.newDLo >= 0) remapped++;
        llvm::errs() << "[postProcessISA] Pass 16: defragmented GEMM2 V-reads"
                     << " (remapped " << remapped << " regs, max v"
                     << (nextFree - 1) << ") at s_setprio 0 line " << scan+1
                     << "\n";
        scan = 0; // restart scan for next body
      }
    }

    if (pass16modified) {
      result.clear();
      for (size_t j = 0; j < allLines.size(); j++) {
        result += allLines[j];
        if (j + 1 < allLines.size()) result += "\n";
      }

      // Update VGPR metadata for newly allocated registers
      auto updateVgprMeta16 = [](std::string &s, const std::string &tag, int newVal) {
        size_t pos = s.find(tag);
        while (pos != std::string::npos) {
          size_t numStart = pos + tag.size();
          while (numStart < s.size() && s[numStart] == ' ') numStart++;
          size_t numEnd = numStart;
          while (numEnd < s.size() && isdigit(s[numEnd])) numEnd++;
          if (numEnd > numStart) {
            int cur = atoi(s.c_str() + numStart);
            if (newVal > cur) {
              std::string nv = std::to_string(newVal);
              s.replace(numStart, numEnd - numStart, nv);
              llvm::errs() << "[Pass 16] Updated " << tag << " from "
                           << cur << " to " << newVal << "\n";
            }
          }
          pos = s.find(tag, numStart + 1);
        }
      };
      // Find max VGPR used by scanning for v[237+] references
      int maxV16 = 0;
      for (int v = 255; v >= 237; v--) {
        std::string vs = "v[" + std::to_string(v) + ":";
        std::string vs2 = ":" + std::to_string(v) + "]";
        if (result.find(vs) != std::string::npos ||
            result.find(vs2) != std::string::npos) {
          maxV16 = v + 1;
          break;
        }
      }
      if (maxV16 > 0) {
        updateVgprMeta16(result, ".amdhsa_next_free_vgpr", maxV16);
        updateVgprMeta16(result, ".vgpr_count:", maxV16);
      }
    }
  }

  // --- Pass 17: DISABLED ---
  // Pre-GEMM1 lgkmcnt(0) CANNOT be relaxed: GFX942 lgkmcnt counter uses
  // strict FIFO ordering. The ds_permute_b32 (issued after 8 K reads) is
  // the 9th/newest LDS operation. Any lgkmcnt(N>0) leaves it outstanding,
  // causing the v_add that reads ds_permute result (v67) to get stale data.
  // Tested lgkmcnt(4) and lgkmcnt(1): both cause MaxErr=2.85e+08.
  // The GEMM2 lgkmcnt(0) instances are also necessary (gate ds_read data
  // for the immediately following MFMA). The compiler already optimized
  // all relaxable lgkmcnt to lgkmcnt(2) or lgkmcnt(3).

  // --- Pass 18: DISABLED ---
  // Yield window (s_nop 15 + s_nop 7 after s_setprio 0) caused regression
  // (114.6T → 113.2T). The 48 extra nop cycles per wave outweigh any
  // barrier stall reduction. Our s_setprio placement (around GEMM1) differs
  // from reference ASM (around O rescale), so the yield timing doesn't help.

  // --- Pass 26: Post-chain handoff spacing knobs ---
  // Keep all Pass 15/15b-15n/25 matcher anchors identical. Only after the full
  // fragile rewrite chain has finished do we adjust a few repeated handoff
  // preludes in the FINAL ISA text.
  //
  // Knobs (read at runtime so multiple variants can be tested without rebuilding):
  //   FLIR_PASS26_DISABLE=1
  //     Skip the pass entirely and keep the pre-Pass26 final ISA for A/B runs.
  //
  //   FLIR_PASS26_LOOPTOP_NOP
  //     Rewrites the already-present loop-top handoff nop block:
  //       s_setprio 0 / s_setprio 1 / s_nop 0
  //     into
  //       s_setprio 0 / s_nop N / s_setprio 1
  //     where N defaults to 0. Match both the legacy
  //     vmcnt(2) loop-top prelude and the split-loop vmcnt(6) variant.
  //
  //   FLIR_PASS26_POSTMFMA_NOP
  //     Inserts a new ASMSTART/s_nop N/ASMEND block after repeated
  //     post-MFMA `s_setprio 0` handoffs whose next region is a SALU+lgkmcnt
  //     prelude followed immediately by another MFMA chain. Disabled by default.
  {
    auto trim = [](const std::string &line) -> std::string {
      size_t start = 0;
      while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
        start++;
      size_t end = line.size();
      while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t'))
        end--;
      return line.substr(start, end - start);
    };
    auto parseEnvInt = [](const char *name, int defaultVal) -> int {
      const char *v = std::getenv(name);
      if (!v || !*v)
        return defaultVal;
      char *end = nullptr;
      long parsed = std::strtol(v, &end, 10);
      if (end == v || (end && *end != '\0'))
        return defaultVal;
      if (parsed < -1)
        return defaultVal;
      if (parsed > 15)
        return defaultVal;
      return static_cast<int>(parsed);
    };
    auto makeAsmBlock = [](const std::string &inst) -> std::vector<std::string> {
      return {"\t;;#ASMSTART", "\t" + inst, "\t;;#ASMEND"};
    };
    auto isLoopTopWait = [&](const std::string &line) -> bool {
      std::string t = trim(line);
      return t == "s_waitcnt vmcnt(2)" || t == "s_waitcnt vmcnt(6)";
    };

    const bool disablePass26 = []() {
      const char *v = std::getenv("FLIR_PASS26_DISABLE");
      if (!v)
        return false;
      std::string s(v);
      for (char &c : s)
        c = static_cast<char>(::tolower(c));
      return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    const int loopTopNop = parseEnvInt("FLIR_PASS26_LOOPTOP_NOP", 0);
    const int postMfmaNop = parseEnvInt("FLIR_PASS26_POSTMFMA_NOP", -1);

    if (!disablePass26) {
      std::vector<std::string> allLines;
      {
        std::istringstream iss(result);
        std::string line;
        while (std::getline(iss, line))
          allLines.push_back(line);
      }

      bool modified = false;
      unsigned loopTopModified = 0;
      unsigned postMfmaModified = 0;
      for (size_t i = 0; i + 10 < allLines.size(); i++) {
        if (trim(allLines[i]) != ";;#ASMSTART" ||
            trim(allLines[i + 1]) != "s_setprio 0" ||
            trim(allLines[i + 2]) != ";;#ASMEND" ||
            trim(allLines[i + 3]) != ";;#ASMSTART" ||
            trim(allLines[i + 4]) != "s_setprio 1" ||
            trim(allLines[i + 5]) != ";;#ASMEND" ||
            trim(allLines[i + 6]) != ";;#ASMSTART" ||
            trim(allLines[i + 7]) != "s_nop 0" ||
            trim(allLines[i + 8]) != ";;#ASMEND" ||
            !isLoopTopWait(allLines[i + 9]) ||
            trim(allLines[i + 10]).find("v_perm_b32") != 0)
          continue;

        std::vector<std::string> sp0Block(allLines.begin() + i,
                                          allLines.begin() + i + 3);
        std::vector<std::string> sp1Block(allLines.begin() + i + 3,
                                          allLines.begin() + i + 6);
        std::vector<std::string> nopBlock(allLines.begin() + i + 6,
                                          allLines.begin() + i + 9);
        nopBlock[1] = "\ts_nop " + std::to_string(loopTopNop);

        std::copy(sp0Block.begin(), sp0Block.end(), allLines.begin() + i);
        std::copy(nopBlock.begin(), nopBlock.end(), allLines.begin() + i + 3);
        std::copy(sp1Block.begin(), sp1Block.end(), allLines.begin() + i + 6);

        llvm::errs() << "[postProcessISA] Pass 26: Retimed loop-top handoff to "
                     << "s_nop " << loopTopNop << " at line " << (i + 1) << "\n";
        modified = true;
        loopTopModified++;
        i += 8;
      }

      if (postMfmaNop >= 0) {
        for (size_t i = 1; i + 2 < allLines.size(); i++) {
          if (trim(allLines[i - 1]).find("v_mfma") != 0 ||
              trim(allLines[i]) != ";;#ASMSTART" ||
              trim(allLines[i + 1]) != "s_setprio 0" ||
              trim(allLines[i + 2]) != ";;#ASMEND")
            continue;

          bool sawLgkmWait = false;
          bool sawNextMfma = false;
          for (size_t j = i + 3; j < std::min(i + 9, allLines.size()); j++) {
            if (trim(allLines[j]) == "s_waitcnt lgkmcnt(0)") {
              sawLgkmWait = true;
              break;
            }
          }
          for (size_t j = i + 3; j < std::min(i + 13, allLines.size()); j++) {
            if (trim(allLines[j]).find("v_mfma") == 0) {
              sawNextMfma = true;
              break;
            }
          }
          if (!sawLgkmWait || !sawNextMfma)
            continue;

          std::vector<std::string> nopBlock =
              makeAsmBlock("s_nop " + std::to_string(postMfmaNop));
          allLines.insert(allLines.begin() + i + 3, nopBlock.begin(),
                          nopBlock.end());
          llvm::errs() << "[postProcessISA] Pass 26: Inserted post-MFMA handoff "
                       << "s_nop " << postMfmaNop << " after line " << (i + 1)
                       << "\n";
          modified = true;
          postMfmaModified++;
          i += 5;
        }
      }

      if (modified) {
        result.clear();
        for (size_t j = 0; j < allLines.size(); j++) {
          result += allLines[j];
          if (j + 1 < allLines.size())
            result += "\n";
        }
      }
      if (loopTopModified || postMfmaModified) {
        llvm::errs() << "[postProcessISA] Pass 26 summary: loopTop="
                     << loopTopModified << ", postMfma=" << postMfmaModified
                     << " (loopTopNop=" << loopTopNop
                     << ", postMfmaNop=" << postMfmaNop << ")\n";
      }
    }
  }

  {
    static int dumpIdx = 0;
    std::string dumpPath =
        "/tmp/postprocess_isa_" + std::to_string(dumpIdx++) + ".s";
    std::error_code ec;
    llvm::raw_fd_ostream dumpFile(dumpPath, ec);
    if (!ec)
      dumpFile << result;
  }

  // Preserve backend-emitted non-zero lgkm waits for the hottest consumer
  // families that post-processing can otherwise flatten to lgkmcnt(0):
  // MFMA, barrier-handoff ds_read bursts, and v_add handoffs after permutes.
  // The earlier text-rewrite passes still operate on their historical
  // lgkmcnt(0) anchors, then we restore the backend count at the very end.
  {
    if (preserveBackendLgkmWaits && !originalLgkmWaits.empty()) {
      std::vector<std::string> allLines = splitIsaLines(result);
      std::vector<std::string> labels;
      std::vector<int> mfmaOrdinals;
      std::vector<int> dsReadOrdinals;
      std::vector<int> vAddOrdinals;
      annotateWaitAnchorLines(allLines, labels, mfmaOrdinals, dsReadOrdinals,
                              vAddOrdinals);

      bool modified = false;
      unsigned restored = 0;
      unsigned restoredMfma = 0;
      unsigned restoredDsRead = 0;
      unsigned restoredVAdd = 0;
      std::set<std::string> restoredKeys;
      auto tryRestoreForAnchor =
          [&](size_t waitIdx, const std::string &kind, const std::string &label,
              int ordinal, const std::string &anchor) -> bool {
        std::string key = makeWaitRestoreKey(kind, label, ordinal, anchor);
        std::string relaxedKey = makeRelaxedWaitRestoreKey(kind, label, anchor);
        auto it = originalLgkmWaits.find(key);
        std::string chosenKey = key;
        if (it == originalLgkmWaits.end() && kind != "mfma") {
          it = originalLgkmWaits.find(relaxedKey);
          chosenKey = relaxedKey;
        }
        if (it == originalLgkmWaits.end() || restoredKeys.count(chosenKey))
          return false;

        size_t lgkmPos = allLines[waitIdx].find("lgkmcnt(");
        if (lgkmPos == std::string::npos)
          return false;
        size_t numStart = lgkmPos + 8;
        size_t numEnd = allLines[waitIdx].find(')', numStart);
        if (numEnd == std::string::npos)
          return false;

        allLines[waitIdx].replace(numStart, numEnd - numStart,
                                  std::to_string(it->second));
        restoredKeys.insert(chosenKey);
        restored++;
        modified = true;
        if (kind == "mfma")
          restoredMfma++;
        else if (kind == "dsread")
          restoredDsRead++;
        else if (kind == "vadd")
          restoredVAdd++;
        return true;
      };
      for (size_t i = 0; i < allLines.size(); i++) {
        if (parseLgkmCount(allLines[i]) != 0 || labels[i].empty())
          continue;

        for (size_t j = i + 1; j < allLines.size() && j <= i + 32; j++) {
          if (labels[j] != labels[i])
            break;
          std::string anchor = trimIsaLine(allLines[j]);
          if (isWaitRestoreAnchorSkippable(anchor))
            continue;

          if (anchor.find("ds_read") == 0 && dsReadOrdinals[j] >= 0) {
            tryRestoreForAnchor(i, "dsread", labels[j], dsReadOrdinals[j],
                                anchor);
            break;
          }
          if (anchor.find("v_add_f32") == 0 && vAddOrdinals[j] >= 0) {
            tryRestoreForAnchor(i, "vadd", labels[j], vAddOrdinals[j], anchor);
            break;
          }
          if (anchor.find("v_mfma") == 0 && mfmaOrdinals[j] >= 0) {
            tryRestoreForAnchor(i, "mfma", labels[j], mfmaOrdinals[j], anchor);
            break;
          }
        }
      }

      if (modified) {
        result.clear();
        for (size_t j = 0; j < allLines.size(); j++) {
          result += allLines[j];
          if (j + 1 < allLines.size())
            result += "\n";
        }
        llvm::errs() << "[postProcessISA] Restored " << restored
                     << " lgkm waits to backend-emitted non-zero counts"
                     << " (mfma=" << restoredMfma
                     << ", dsread=" << restoredDsRead
                     << ", vadd=" << restoredVAdd << ")\n";
      }
    }
  }

  // --- Pass 28: Minimal loop back-edge phase offset ---
  // Final dual-group attempt: keep the whole frontend and fragile Pass
  // 15/25/26 chain unchanged, and only add a tiny fixed delay at the loop
  // back-edge. The intent is to create a natural inter-wave phase offset
  // without changing priorities, register pressure, or control flow.
  {
    auto parseBackedgeNop = [](const char *name, int defaultVal) -> int {
      const char *v = std::getenv(name);
      if (!v || !*v)
        return defaultVal;
      char *end = nullptr;
      long parsed = std::strtol(v, &end, 10);
      if (end == v || (end && *end != '\0'))
        return defaultVal;
      if (parsed < 0 || parsed > 4)
        return defaultVal;
      return static_cast<int>(parsed);
    };
    const bool disablePass28 = []() {
      const char *v = std::getenv("FLIR_PASS28_DISABLE");
      if (!v)
        return false;
      std::string s(v);
      for (char &c : s)
        c = static_cast<char>(::tolower(c));
      return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    const int backedgeNop = parseBackedgeNop("FLIR_PASS28_BACKEDGE_NOP", 1);

    if (!disablePass28 && backedgeNop > 0) {
      std::vector<std::string> allLines = splitIsaLines(result);
      std::map<std::string, size_t> firstLabelLine;

      for (size_t i = 0; i < allLines.size(); ++i) {
        std::string t = trimIsaLine(allLines[i]);
        if (t.empty() || t[0] != '.')
          continue;
        size_t colonPos = t.find(':');
        if (colonPos == std::string::npos)
          continue;
        std::string label = t.substr(0, colonPos);
        firstLabelLine.emplace(label, i);
      }

      bool modified = false;
      unsigned inserted = 0;
      for (size_t i = 0; i < allLines.size(); ++i) {
        std::string t = trimIsaLine(allLines[i]);
        if (!(t.find("s_cbranch_vccnz ") == 0 || t.find("s_branch ") == 0))
          continue;

        size_t spacePos = t.find(' ');
        if (spacePos == std::string::npos)
          continue;
        std::string target = t.substr(spacePos + 1);
        auto labelIt = firstLabelLine.find(target);
        if (labelIt == firstLabelLine.end() || labelIt->second >= i)
          continue;

        if (i > 0 && trimIsaLine(allLines[i - 1]) ==
                         ("s_nop " + std::to_string(backedgeNop)))
          continue;

        allLines.insert(allLines.begin() + i,
                        "\ts_nop " + std::to_string(backedgeNop));
        llvm::errs() << "[postProcessISA] Pass 28: Inserted loop back-edge "
                     << "s_nop " << backedgeNop << " before " << t
                     << " at line " << (i + 1) << "\n";
        modified = true;
        inserted++;
        i++;
      }

      if (modified) {
        result.clear();
        for (size_t j = 0; j < allLines.size(); ++j) {
          result += allLines[j];
          if (j + 1 < allLines.size())
            result += "\n";
        }
        llvm::errs() << "[postProcessISA] Pass 28 summary: inserted="
                     << inserted << " (backedgeNop=" << backedgeNop << ")\n";
      }
    }
  }

  return result;
}

std::optional<SmallVector<char, 0>> SerializeGPUModuleBase::moduleToObjectImpl(
    const gpu::TargetOptions &targetOptions, llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is offload.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: "
                 << cast<gpu::GPUModuleOp>(getOperation()).getNameAttr() << "\n"
                 << llvmModule << "\n";
  });
#undef DEBUG_TYPE
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

  // Apply cmdOptions as LLVM command-line flags so they reach the AMDGPU
  // backend's scheduling and waitcnt-insertion passes.
  // Filter out unregistered options to avoid ParseCommandLineOptions aborting
  // on the first unknown flag and skipping all subsequent valid flags.
  {
    auto cmdOpts = targetOptions.tokenizeCmdOptions();
    if (!cmdOpts.second.empty()) {
      auto &registeredOpts =
          llvm::cl::getRegisteredOptions(llvm::cl::SubCommand::getTopLevel());
      SmallVector<const char *, 16> argv;
      argv.push_back("mlir-rocdl");
      for (const char *opt : cmdOpts.second) {
        StringRef s(opt);
        StringRef name = s;
        if (name.starts_with("--"))
          name = name.drop_front(2);
        else if (name.starts_with("-"))
          name = name.drop_front(1);
        name = name.split('=').first;
        if (registeredOpts.count(name)) {
          argv.push_back(opt);
        } else {
          llvm::errs() << "ROCDL: skipping unregistered option: " << s << "\n";
        }
      }
      if (argv.size() > 1) {
        llvm::cl::ResetAllOptionOccurrences();
        std::string parseErrs;
        llvm::raw_string_ostream errStream(parseErrs);
        if (!llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(),
                                               "ROCDL LLVM backend options\n",
                                               &errStream)) {
          llvm::errs() << "Warning: LLVM opts parse error: " << parseErrs
                       << "\n";
        }
      }
    }
  }

  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "target Machine unavailable for triple "
                               << triple << ", can't compile with LLVM";
    return std::nullopt;
  }

  // Translate the Module to ISA.
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, **targetMachine);
  if (!serializedISA) {
    getOperation().emitError() << "failed translating the module to ISA";
    return std::nullopt;
  }

  // Post-process ISA: insert yield nops after s_setprio 0 in inner loops.
  *serializedISA = postProcessISA(*serializedISA);

  // Debug: dump modified ISA to temp file
  {
    static int dumpIdx = 0;
    const char *targetName = "unknown";
    switch (targetOptions.getCompilationTarget()) {
    case gpu::CompilationTarget::Assembly: targetName = "isa"; break;
    case gpu::CompilationTarget::Binary: targetName = "bin"; break;
    case gpu::CompilationTarget::Fatbin: targetName = "fatbin"; break;
    default: break;
    }
    std::string path = "/tmp/postprocess_isa_" + std::to_string(dumpIdx++) + ".s";
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (!ec) os << *serializedISA;
    llvm::errs() << "[moduleToObjectImpl] target=" << targetName
                 << " dump=" << path
                 << " size=" << serializedISA->size() << "\n";
  }

#define DEBUG_TYPE "serialize-to-isa"
  LLVM_DEBUG({
    llvm::dbgs() << "ISA for module: "
                 << cast<gpu::GPUModuleOp>(getOperation()).getNameAttr() << "\n"
                 << *serializedISA << "\n";
  });
#undef DEBUG_TYPE
  // Return ISA assembly code if the compilation target is assembly.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Assembly)
    return SmallVector<char, 0>(serializedISA->begin(), serializedISA->end());

  // Compiling to binary requires a valid ROCm path, fail if it's not found.
  if (getToolkitPath().empty()) {
    getOperation().emitError() << "invalid ROCm path, please set a valid path";
    return std::nullopt;
  }

  // Compile to binary.
  return compileToBinary(*serializedISA);
}

#if MLIR_ENABLE_ROCM_CONVERSIONS
namespace {
class AMDGPUSerializer : public SerializeGPUModuleBase {
public:
  AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                   const gpu::TargetOptions &targetOptions);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  // Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

AMDGPUSerializer::AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                                   const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {}

std::optional<SmallVector<char, 0>>
AMDGPUSerializer::moduleToObject(llvm::Module &llvmModule) {
  return moduleToObjectImpl(targetOptions, llvmModule);
}
#endif // MLIR_ENABLE_ROCM_CONVERSIONS

std::optional<SmallVector<char, 0>> ROCDLTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("module must be a GPU module");
    return std::nullopt;
  }
#if MLIR_ENABLE_ROCM_CONVERSIONS
  AMDGPUSerializer serializer(*module, cast<ROCDLTargetAttr>(attribute),
                              options);
  serializer.init();
  return serializer.run();
#else
  module->emitError("the `AMDGPU` target was not built. Please enable it when "
                    "building LLVM");
  return std::nullopt;
#endif // MLIR_ENABLE_ROCM_CONVERSIONS
}

Attribute
ROCDLTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                  const SmallVector<char, 0> &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  // If format is `fatbin` transform it to binary as `fatbin` is not yet
  // supported.
  gpu::KernelTableAttr kernels;
  if (format > gpu::CompilationTarget::Binary) {
    format = gpu::CompilationTarget::Binary;
    kernels = ROCDL::getKernelMetadata(module, object);
  }
  DictionaryAttr properties{};
  Builder builder(attribute.getContext());
  StringAttr objectStr =
      builder.getStringAttr(StringRef(object.data(), object.size()));
  return builder.getAttr<gpu::ObjectAttr>(attribute, format, objectStr,
                                          properties, kernels);
}
