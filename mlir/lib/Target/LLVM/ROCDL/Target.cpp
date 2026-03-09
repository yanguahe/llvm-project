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

  // --- Pass 4d: Remove redundant consecutive lgkmcnt(0) ---
  // Only remove if the second lgkmcnt(0) is within 200 chars of the first
  // (i.e., truly consecutive with only a few VALU ops between).
  // Also reject if the second lgkmcnt(0) is immediately before s_barrier
  // (pre-barrier waits must be kept even if technically redundant).
  unsigned lgkm0Removed = 0;
  {
    const std::string lk0 = "lgkmcnt(0)";
    size_t pos = 0;
    while (true) {
      size_t first = loop.find(lk0, pos);
      if (first == std::string::npos) break;
      size_t firstEnd = loop.find('\n', first);
      if (firstEnd == std::string::npos) break;
      firstEnd++;
      size_t second = loop.find(lk0, firstEnd);
      if (second == std::string::npos) break;
      size_t gap = second - firstEnd;
      std::string between = loop.substr(firstEnd, gap);
      if (gap < 200 &&
          between.find("ds_read") == std::string::npos &&
          between.find("ds_write") == std::string::npos &&
          between.find("ds_permute") == std::string::npos &&
          between.find("buffer_load") == std::string::npos &&
          between.find("s_barrier") == std::string::npos) {
        // Don't remove if second lgkmcnt(0) is right before s_barrier
        size_t secLineEnd = loop.find('\n', second);
        secLineEnd = (secLineEnd == std::string::npos) ? loop.size() : secLineEnd + 1;
        std::string afterSec = loop.substr(secLineEnd,
            std::min((size_t)40, loop.size() - secLineEnd));
        if (afterSec.find("s_barrier") != std::string::npos) {
          pos = firstEnd;
          continue;
        }
        size_t secLineStart = loop.rfind('\n', second);
        secLineStart = (secLineStart == std::string::npos) ? 0 : secLineStart + 1;
        std::string secLine = loop.substr(secLineStart, second + 15 - secLineStart);
        if (secLine.find("expcnt") == std::string::npos) {
          loop.erase(secLineStart, secLineEnd - secLineStart);
          lgkm0Removed++;
          llvm::errs() << "[postProcessISA] Removed redundant lgkmcnt(0) in "
                       << label << "\n";
          continue;
        }
      }
      pos = firstEnd;
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

  llvm::errs() << "[postProcessISA] " << label << ": moved " << cmpMoved
               << " v_cmp, yield=" << yieldInserted
               << ", nopFilled=" << nopFilled
               << ", vmcntRelaxed=" << vmcntRelaxed
               << ", vmcnt0Removed=" << vmcnt0Removed
               << ", lgkm0Removed=" << lgkm0Removed
               << ", lgkmRelaxed=" << lgkmRelaxed
               << ", fullWaitRemoved=" << fullWaitRemoved
               << ", setprioRemoved=" << setprioRemoved << "\n";

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
    unsigned loopBarriers = 0;
    {
      size_t bpos = 0;
      while (true) {
        size_t b = loop.find("s_barrier", bpos);
        if (b == std::string::npos) break;
        loopBarriers++;
        bpos = b + 10;
      }
    }
    bool doYield = (loopBarriers > 2);
    bool fillGap = false;
    bool hoistVcmp = (loopBarriers <= 2);
    loop = processOneLoop(std::move(loop), label, doYield, fillGap, hoistVcmp);

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

    // --- Pass 10: Scalar branch to skip causal mask for non-boundary blocks ---
    // For "Body A" style mask blocks (v_cmp + v_cndmask + v_max3 interleaved,
    // with v_cndmask writing back to the SAME register), add a scalar branch
    // that jumps to a fast path doing only the v_max3 rowmax when no lane
    // needs masking (mask_delta > max_threshold for all active lanes).
    unsigned maskBranchesAdded = 0;
    if (loopBarriers > 2) {
      size_t mSearchPos = 0;
      while (maskBranchesAdded < 1) {
        // Find start of a Body-A mask block: v_cmp_lt_i32_e64 s[0:1], ...
        size_t firstCmp = loop.find("v_cmp_lt_i32_e64 s[0:1], ", mSearchPos);
        if (firstCmp == std::string::npos) break;

        size_t firstCmpLineStart = loop.rfind('\n', firstCmp);
        firstCmpLineStart = (firstCmpLineStart == std::string::npos)
                                ? 0 : firstCmpLineStart + 1;
        size_t firstCmpLineEnd = loop.find('\n', firstCmp);
        if (firstCmpLineEnd == std::string::npos) {
          mSearchPos = firstCmp + 1; continue;
        }

        // Extract mask_delta register (last token: "v_cmp_lt_i32_e64 s[0:1], <th>, v<N>")
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

        // Count v_cmp_lt_i32 comparing against maskDeltaReg and find max threshold
        int cmpCount = 0;
        int maxThreshold = -100;
        size_t scanLim = std::min(firstCmpLineStart + 1500, loop.size());
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
            // Extract threshold: "v_cmp_lt_i32_e64 s[N:N+1], <th>, v<M>"
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
          }
          if (cl.find("v_mfma") != std::string::npos ||
              cl.find("s_barrier") != std::string::npos) break;
          pos = le + 1;
        }

        if (cmpCount < 14 || maxThreshold < 20) {
          mSearchPos = firstCmpLineEnd + 1; continue;
        }

        // Collect v_max3_f32 instructions in the mask+rowmax block.
        // Body A pattern: v_max3 writes to maskDeltaReg, interleaved with v_cndmask.
        // Stop at s_waitcnt or ds_permute (past mask section).
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

        // --- Fallback: v_max3 not interleaved with v_cndmask ---
        // In the current ISA layout, v_max3 appears AFTER MFMAs (not in the
        // v_cmp+v_cndmask region). We skip the v_cmp+v_cndmask block entirely.
        // Any non-mask instructions (buffer_load, s_mov, etc.) inside the
        // skipped region are duplicated in the fast path to maintain correctness.
        if (max3Lines.size() < 7) {
          size_t lastCndEnd = 0;
          int cndCount = 0;
          std::vector<std::string> sideEffectLines;
          pos = firstCmpLineStart;
          scanLim = std::min(firstCmpLineStart + 2500, loop.size());
          while (pos < scanLim) {
            size_t le = loop.find('\n', pos);
            if (le == std::string::npos) break;
            std::string cl = loop.substr(pos, le - pos);
            if (cl.find("v_cndmask_b32_e64") != std::string::npos) {
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

          if (cndCount < 14 || lastCndEnd == 0) {
            mSearchPos = firstCmpLineEnd + 1; continue;
          }

          std::string fastLbl = ".Lfm_" + label.substr(1) + "_" +
                                std::to_string(maskBranchesAdded);
          std::string mergeLbl = ".Lmm_" + label.substr(1) + "_" +
                                 std::to_string(maskBranchesAdded);

          std::string checkCode;
          checkCode += "\tv_cmp_ge_i32_e64 s[0:1], " +
                       std::to_string(maxThreshold) + ", " +
                       maskDeltaReg + "\n";
          checkCode += "\ts_and_b64 s[0:1], s[0:1], exec\n";
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

          llvm::errs() << "[postProcessISA] Added mask skip (fallback) in " << label
                       << " body " << maskBranchesAdded
                       << " (cmp=" << cmpCount
                       << ", cnd=" << cndCount
                       << ", sideEffects=" << sideEffectLines.size()
                       << ", maxTh=" << maxThreshold
                       << ", delta=" << maskDeltaReg << ")\n";

          maskBranchesAdded++;
          mSearchPos = lastCndEnd + fastCode.size();
          continue;
        }

        // Verify this is a Body-A block: v_max3 should appear within 600 chars
        // of the last v_cmp (Body B has MFMA between v_cmp and v_max3).
        size_t firstMax3Pos = loop.find("v_max3_f32", lastCmpLineEnd);
        if (firstMax3Pos == std::string::npos ||
            firstMax3Pos - lastCmpLineEnd > 600) {
          mSearchPos = firstCmpLineEnd + 1; continue;
        }

        // Build unique labels
        std::string fastLbl = ".Lfm_" + label.substr(1) + "_" +
                              std::to_string(maskBranchesAdded);
        std::string mergeLbl = ".Lmm_" + label.substr(1) + "_" +
                               std::to_string(maskBranchesAdded);

        // --- Insert branch check before firstCmpLineStart ---
        // v_cmp_ge_i32_e64 s[0:1], <maxTh>, <maskDelta>  ; lanes needing mask
        // s_and_b64 s[0:1], s[0:1], exec                  ; SCC=1 if any
        // s_cbranch_scc0 <fastLbl>                         ; skip if none
        std::string checkCode;
        checkCode += "\tv_cmp_ge_i32_e64 s[0:1], " +
                     std::to_string(maxThreshold) + ", " +
                     maskDeltaReg + "\n";
        checkCode += "\ts_and_b64 s[0:1], s[0:1], exec\n";
        checkCode += "\ts_cbranch_scc0 " + fastLbl + "\n";

        loop.insert(firstCmpLineStart, checkCode);
        size_t checkLen = checkCode.size();
        lastMax3End += checkLen;

        // --- Insert fast path after lastMax3End ---
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

    // --- Pass 11: Relocate s_setprio around O rescale + add yield NOPs ---
    // DISABLED: Relocating s_setprio from GEMM1 boundary (compiler-chosen)
    // to O rescale boundary (reference ASM pattern) caused regression
    // (111T → 110.4T without yield, 107.7T with yield). The 2-body loop
    // structure has different timing than the reference's 5-body loop.

    // --- Pass 12: Add yield NOPs after s_setprio 0 ---
    // DISABLED: Yield NOPs (s_nop 15 + s_nop 7) after s_setprio 0 caused
    // regression (111T → 109T). The 2-body loop doesn't benefit from the
    // reference ASM's yield pattern.

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

  {
    static int dumpIdx = 0;
    std::string dumpPath =
        "/tmp/postprocess_isa_" + std::to_string(dumpIdx++) + ".s";
    std::error_code ec;
    llvm::raw_fd_ostream dumpFile(dumpPath, ec);
    if (!ec)
      dumpFile << result;
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
