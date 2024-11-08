#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "polly/Support/GICHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "GPULowering.h"
#include "Utils.h"

using namespace mlir;

namespace {

Type convertMemrefElementTypeForLLVMPointer(
    MemRefType type, const LLVMTypeConverter &converter) {
  Type converted = converter.convertType(type.getElementType());
  if (!converted)
    return Type();

  if (type.getRank() == 0) {
    return converted;
  }

  // Only the leading dimension can be dynamic.
  if (llvm::any_of(type.getShape().drop_front(), ShapedType::isDynamic))
    return Type();

  // Only identity layout is supported.
  // TODO: detect the strided layout that is equivalent to identity
  // given the static part of the shape.
  if (!type.getLayout().isIdentity())
    return Type();

  if (type.getRank() > 0) {
    for (int64_t size : llvm::reverse(type.getShape().drop_front()))
      converted = LLVM::LLVMArrayType::get(converted, size);
  }
  return converted;
}

/// Converts the given memref type into the LLVM type that can be used for a
/// global. The memref type must have all dimensions statically known. The
/// provided type converter is used to convert the elemental type.
static Type convertGlobalMemRefTypeToLLVM(MemRefType type,
                                          const TypeConverter &typeConverter) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return nullptr;

  Type convertedType = typeConverter.convertType(type.getElementType());
  if (!convertedType)
    return nullptr;

  for (int64_t size : llvm::reverse(type.getShape()))
    convertedType = LLVM::LLVMArrayType::get(convertedType, size);
  return convertedType;
}

struct GetGlobalOpLowering
    : public ConvertOpToLLVMPattern<memref::GetGlobalOp> {
public:
  using ConvertOpToLLVMPattern<memref::GetGlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType originalType = getGlobalOp.getType();
    Type convertedType = getTypeConverter()->convertType(originalType);
    Value wholeAddress = rewriter.create<LLVM::AddressOfOp>(
        getGlobalOp->getLoc(), convertedType, getGlobalOp.getName());

    rewriter.replaceOp(getGlobalOp, wholeAddress);
    return success();
  }
};

struct GlobalOpLowering : public ConvertOpToLLVMPattern<memref::GlobalOp> {
public:
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType originalType = globalOp.getType();
    if (!originalType.hasStaticShape() ||
        !originalType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(globalOp->getLoc(),
                                         "unsupported type");
    }

    Type convertedType =
        convertGlobalMemRefTypeToLLVM(originalType, *typeConverter);
    LLVM::Linkage linkage =
        globalOp.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;
    if (!convertedType) {
      return rewriter.notifyMatchFailure(globalOp->getLoc(),
                                         "failed to convert memref type");
    }

    Attribute initialValue = nullptr;
    if (!globalOp.isExternal() && !globalOp.isUninitialized()) {
      auto elementsAttr = globalOp.getInitialValue()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (originalType.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }

    IntegerAttr alignment = globalOp.getAlignmentAttr();
    bool dso_local = globalOp->getAttr("polygeist.cuda_device") ||
                     globalOp->getAttr("polygeist.cuda_constant");
    bool thread_local_ = false;
    LLVM::UnnamedAddrAttr unnamed_addr = nullptr;
    StringAttr section = nullptr;
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        globalOp, convertedType, globalOp.getConstant(), globalOp.getSymName(),
        linkage, dso_local, thread_local_, initialValue, alignment,
        originalType.getMemorySpaceAsInt(), unnamed_addr, section,
        /*comdat=*/nullptr, /*dbg_expr=*/nullptr);
    if (!globalOp.isExternal() && globalOp.isUninitialized()) {
      Block *block =
          rewriter.createBlock(&newGlobal.getInitializerRegion(),
                               newGlobal.getInitializerRegion().begin());
      rewriter.setInsertionPointToStart(block);
      Value undef =
          rewriter.create<LLVM::UndefOp>(globalOp->getLoc(), convertedType);
      rewriter.create<LLVM::ReturnOp>(globalOp->getLoc(), undef);
    }
    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(mt))
      return rewriter.notifyMatchFailure(ao, "Not shared memory address space");

    auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                /* memspace */ 3);
    auto loc = ao->getLoc();
    auto name = "shared_mem_" + std::to_string((long long int)(Operation *)ao);

    auto mod = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!mod) {
      return failure();
    }

    rewriter.setInsertionPointToStart(mod.getBody());

    auto initialValue = rewriter.getUnitAttr();
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name),
        /* sym_visibility */ mlir::StringAttr(), mlir::TypeAttr::get(type),
        initialValue, mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

/// Pattern for lowering automatic stack allocations.
/// Pattern for allocation-like operations.
template <typename OpTy>
struct AllocLikeOpLowering : public ConvertOpToLLVMPattern<OpTy> {
public:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

protected:
  /// Returns the value containing the outermost dimension of the memref to be
  /// allocated, or 1 if the memref has rank zero.
  Value getOuterSize(OpTy original,
                     typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (!adaptor.getDynamicSizes().empty())
      return adaptor.getDynamicSizes().front();

    // TODO index size
    Type indexType = rewriter.getI64Type();
    return this->createIndexAttrConstant(
        rewriter, original->getLoc(), indexType,
        original.getType().getRank() == 0 ? 1
                                          : original.getType().getDimSize(0));
  }
};

/// Pattern for lowering automatic stack allocations.
struct CAllocaOpLowering : public AllocLikeOpLowering<memref::AllocaOp> {
public:
  using AllocLikeOpLowering<memref::AllocaOp>::AllocLikeOpLowering;

  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocaOp.getLoc();
    MemRefType originalType = allocaOp.getType();
    if (!originalType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(allocaOp,
                                         "Memref layout is not identity");

    if (!originalType.hasStaticShape())
      return rewriter.notifyMatchFailure(allocaOp, "Alloca with dynamic sizes");
    auto convertedType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(originalType));
    auto elTy = convertMemrefElementTypeForLLVMPointer(
        originalType, *this->getTypeConverter());
    if (!convertedType || !elTy)
      return rewriter.notifyMatchFailure(loc, "unsupported memref type");

    assert(adaptor.getDynamicSizes().size() <= 1 &&
           "expected at most one dynamic size");

    Value outerSize = getOuterSize(allocaOp, adaptor, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        allocaOp, convertedType, elTy, outerSize,
        adaptor.getAlignment().value_or(0));
    return success();
  }
};

struct AtAddrLower : public ConvertOpToLLVMPattern<memref::AtAddrOp> {
  using ConvertOpToLLVMPattern<memref::AtAddrOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(memref::AtAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, ValueRange({adaptor.getAddr()}));
    return success();
  }
};

struct VectorLoadLower : public ConvertOpToLLVMPattern<vector::LoadOp> {
  using ConvertOpToLLVMPattern<vector::LoadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = cast_or_null<TypeAttr>(op->getAttr("polymer.access.type"));
    if (!tyAttr)
      return rewriter.notifyMatchFailure(op, "Access type attribute missing");
    auto memref = op.getBase();
    if (!memref.getType().getLayout().isIdentity())
      return rewriter.notifyMatchFailure(op, "Memref layout is not identity");

    Type ty = tyAttr.getValue();
    Value ptr = adaptor.getBase();

    Value newVal = bitcastToVec(
        rewriter, getTypeConverter()->getDataLayoutAnalysis()->getAbove(op),

        rewriter.create<LLVM::LoadOp>(op.getLoc(), ty, ptr));

    rewriter.replaceOp(op, newVal);

    return success();
  }
};

struct VectorStoreLower : public ConvertOpToLLVMPattern<vector::StoreOp> {
  using ConvertOpToLLVMPattern<vector::StoreOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(vector::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = cast_or_null<TypeAttr>(op->getAttr("polymer.access.type"));
    if (!tyAttr)
      return rewriter.notifyMatchFailure(op, "Access type attribute missing");
    auto memref = op.getBase();
    if (!memref.getType().getLayout().isIdentity())
      return rewriter.notifyMatchFailure(op, "Memref layout is not identity");

    Type ty = tyAttr.getValue();
    Value ptr = adaptor.getBase();

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op,
        bitcastFromVec(
            rewriter, getTypeConverter()->getDataLayoutAnalysis()->getAbove(op),
            ty, adaptor.getValueToStore()),
        ptr);

    return success();
  }
};

template <typename OpTy>
struct CLoadStoreOpLowering : public ConvertOpToLLVMPattern<OpTy> {
protected:
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  /// Emits the IR that computes the address of the memory being accessed.
  Value getAddress(OpTy op,
                   typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MemRefType originalType = op.getMemRefType();
    auto convertedType = dyn_cast_or_null<LLVM::LLVMPointerType>(
        this->getTypeConverter()->convertType(originalType));
    if (!convertedType) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return nullptr;
    }

    SmallVector<LLVM::GEPArg> args = llvm::to_vector(llvm::map_range(
        adaptor.getIndices(), [](Value v) { return LLVM::GEPArg(v); }));
    auto elTy = convertMemrefElementTypeForLLVMPointer(
        originalType, *this->getTypeConverter());
    if (!elTy) {
      (void)rewriter.notifyMatchFailure(loc, "unsupported memref type");
      return nullptr;
    }
    return rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(op.getContext(),
                                   originalType.getMemorySpaceAsInt()),
        elTy, adaptor.getMemref(), args);
  }
};

struct CLoadOpLowering : public CLoadStoreOpLowering<memref::LoadOp> {
public:
  using CLoadStoreOpLowering<memref::LoadOp>::CLoadStoreOpLowering;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value address = getAddress(loadOp, adaptor, rewriter);
    if (!address)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp,
        typeConverter->convertType(loadOp.getMemRefType().getElementType()),
        address);
    return success();
  }
};

struct CStoreOpLowering : public CLoadStoreOpLowering<memref::StoreOp> {
public:
  using CLoadStoreOpLowering<memref::StoreOp>::CLoadStoreOpLowering;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value address = getAddress(storeOp, adaptor, rewriter);
    if (!address)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.getValue(),
                                               address);
    return success();
  }
};

} // namespace

void mlir::populateGPULoweringPatterns(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter) {
  patterns.add<SharedMemrefAllocaToGlobal>(patterns.getContext());
  patterns.add<AtAddrLower, CLoadOpLowering, CStoreOpLowering, VectorStoreLower,
               VectorLoadLower, CAllocaOpLowering, GlobalOpLowering,
               GetGlobalOpLowering>(typeConverter);
}
