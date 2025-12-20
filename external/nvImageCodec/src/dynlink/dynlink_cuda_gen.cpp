#include <cuda.h>

void* CudaLoadSymbol(const char* name);

#define LOAD_SYMBOL_FUNC Cuda##LoadSymbol

CUresult CUDAAPI cuGetErrorStringNotFound(CUresult, const char**) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    using FuncPtr = CUresult (*)(CUresult, const char**);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuGetErrorString")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuGetErrorString")) : cuGetErrorStringNotFound;
    return func_ptr(error, pStr);
}

CUresult CUDAAPI cuGetErrorNameNotFound(CUresult, const char**) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    using FuncPtr = CUresult (*)(CUresult, const char**);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuGetErrorName")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuGetErrorName")) : cuGetErrorNameNotFound;
    return func_ptr(error, pStr);
}

CUresult CUDAAPI cuInitNotFound(unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuInit(unsigned int Flags) {
    using FuncPtr = CUresult (*)(unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuInit")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuInit")) : cuInitNotFound;
    return func_ptr(Flags);
}

CUresult CUDAAPI cuDriverGetVersionNotFound(int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    using FuncPtr = CUresult (*)(int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDriverGetVersion")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDriverGetVersion")) : cuDriverGetVersionNotFound;
    return func_ptr(driverVersion);
}

CUresult CUDAAPI cuDeviceGetNotFound(CUdevice*, int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    using FuncPtr = CUresult (*)(CUdevice*, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGet")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGet")) : cuDeviceGetNotFound;
    return func_ptr(device, ordinal);
}

CUresult CUDAAPI cuDeviceGetCountNotFound(int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceGetCount(int* count) {
    using FuncPtr = CUresult (*)(int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetCount")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetCount")) : cuDeviceGetCountNotFound;
    return func_ptr(count);
}

CUresult CUDAAPI cuDeviceGetNameNotFound(char*, int, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    using FuncPtr = CUresult (*)(char*, int, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetName")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetName")) : cuDeviceGetNameNotFound;
    return func_ptr(name, len, dev);
}

CUresult CUDAAPI cuDeviceTotalMem_v2NotFound(size_t*, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
    using FuncPtr = CUresult (*)(size_t*, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceTotalMem_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceTotalMem_v2")) : cuDeviceTotalMem_v2NotFound;
    return func_ptr(bytes, dev);
}

CUresult CUDAAPI cuDeviceGetAttributeNotFound(int*, CUdevice_attribute, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    using FuncPtr = CUresult (*)(int*, CUdevice_attribute, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetAttribute")) : cuDeviceGetAttributeNotFound;
    return func_ptr(pi, attrib, dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxRetainNotFound(CUcontext*, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    using FuncPtr = CUresult (*)(CUcontext*, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxRetain")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxRetain")) : cuDevicePrimaryCtxRetainNotFound;
    return func_ptr(pctx, dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2NotFound(CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    using FuncPtr = CUresult (*)(CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxRelease_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxRelease_v2")) : cuDevicePrimaryCtxRelease_v2NotFound;
    return func_ptr(dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2NotFound(CUdevice, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    using FuncPtr = CUresult (*)(CUdevice, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxSetFlags_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxSetFlags_v2")) : cuDevicePrimaryCtxSetFlags_v2NotFound;
    return func_ptr(dev, flags);
}

CUresult CUDAAPI cuDevicePrimaryCtxGetStateNotFound(CUdevice, unsigned int*, int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    using FuncPtr = CUresult (*)(CUdevice, unsigned int*, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxGetState")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxGetState")) : cuDevicePrimaryCtxGetStateNotFound;
    return func_ptr(dev, flags, active);
}

CUresult CUDAAPI cuDevicePrimaryCtxReset_v2NotFound(CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    using FuncPtr = CUresult (*)(CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxReset_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDevicePrimaryCtxReset_v2")) : cuDevicePrimaryCtxReset_v2NotFound;
    return func_ptr(dev);
}

CUresult CUDAAPI cuCtxCreate_v2NotFound(CUcontext*, unsigned int, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    using FuncPtr = CUresult (*)(CUcontext*, unsigned int, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxCreate_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxCreate_v2")) : cuCtxCreate_v2NotFound;
    return func_ptr(pctx, flags, dev);
}

CUresult CUDAAPI cuCtxCreate_v4NotFound(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) {
    using FuncPtr = CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxCreate_v4")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxCreate_v4")) : cuCtxCreate_v4NotFound;
    return func_ptr(pctx, ctxCreateParams, flags, dev);
}

CUresult CUDAAPI cuCtxDestroy_v2NotFound(CUcontext) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    using FuncPtr = CUresult (*)(CUcontext);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxDestroy_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxDestroy_v2")) : cuCtxDestroy_v2NotFound;
    return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxPushCurrent_v2NotFound(CUcontext) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    using FuncPtr = CUresult (*)(CUcontext);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxPushCurrent_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxPushCurrent_v2")) : cuCtxPushCurrent_v2NotFound;
    return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxPopCurrent_v2NotFound(CUcontext*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
    using FuncPtr = CUresult (*)(CUcontext*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxPopCurrent_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxPopCurrent_v2")) : cuCtxPopCurrent_v2NotFound;
    return func_ptr(pctx);
}

CUresult CUDAAPI cuCtxSetCurrentNotFound(CUcontext) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    using FuncPtr = CUresult (*)(CUcontext);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetCurrent")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetCurrent")) : cuCtxSetCurrentNotFound;
    return func_ptr(ctx);
}

CUresult CUDAAPI cuCtxGetCurrentNotFound(CUcontext*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    using FuncPtr = CUresult (*)(CUcontext*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetCurrent")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetCurrent")) : cuCtxGetCurrentNotFound;
    return func_ptr(pctx);
}

CUresult CUDAAPI cuCtxGetDeviceNotFound(CUdevice*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    using FuncPtr = CUresult (*)(CUdevice*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetDevice")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetDevice")) : cuCtxGetDeviceNotFound;
    return func_ptr(device);
}

CUresult CUDAAPI cuCtxGetFlagsNotFound(unsigned int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetFlags(unsigned int* flags) {
    using FuncPtr = CUresult (*)(unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetFlags")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetFlags")) : cuCtxGetFlagsNotFound;
    return func_ptr(flags);
}

CUresult CUDAAPI cuCtxSynchronizeNotFound() {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxSynchronize() {
    using FuncPtr = CUresult (*)();

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSynchronize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSynchronize")) : cuCtxSynchronizeNotFound;
    return func_ptr();
}

CUresult CUDAAPI cuCtxSetLimitNotFound(CUlimit, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    using FuncPtr = CUresult (*)(CUlimit, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetLimit")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetLimit")) : cuCtxSetLimitNotFound;
    return func_ptr(limit, value);
}

CUresult CUDAAPI cuCtxGetLimitNotFound(size_t*, CUlimit) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
    using FuncPtr = CUresult (*)(size_t*, CUlimit);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetLimit")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetLimit")) : cuCtxGetLimitNotFound;
    return func_ptr(pvalue, limit);
}

CUresult CUDAAPI cuCtxGetCacheConfigNotFound(CUfunc_cache*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig) {
    using FuncPtr = CUresult (*)(CUfunc_cache*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetCacheConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetCacheConfig")) : cuCtxGetCacheConfigNotFound;
    return func_ptr(pconfig);
}

CUresult CUDAAPI cuCtxSetCacheConfigNotFound(CUfunc_cache) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
    using FuncPtr = CUresult (*)(CUfunc_cache);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetCacheConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetCacheConfig")) : cuCtxSetCacheConfigNotFound;
    return func_ptr(config);
}

CUresult CUDAAPI cuCtxGetApiVersionNotFound(CUcontext, unsigned int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
    using FuncPtr = CUresult (*)(CUcontext, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetApiVersion")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetApiVersion")) : cuCtxGetApiVersionNotFound;
    return func_ptr(ctx, version);
}

CUresult CUDAAPI cuCtxGetStreamPriorityRangeNotFound(int*, int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    using FuncPtr = CUresult (*)(int*, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetStreamPriorityRange")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetStreamPriorityRange")) : cuCtxGetStreamPriorityRangeNotFound;
    return func_ptr(leastPriority, greatestPriority);
}

CUresult CUDAAPI cuCtxGetSharedMemConfigNotFound(CUsharedconfig*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) {
    using FuncPtr = CUresult (*)(CUsharedconfig*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetSharedMemConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxGetSharedMemConfig")) : cuCtxGetSharedMemConfigNotFound;
    return func_ptr(pConfig);
}

CUresult CUDAAPI cuCtxSetSharedMemConfigNotFound(CUsharedconfig) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
    using FuncPtr = CUresult (*)(CUsharedconfig);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetSharedMemConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxSetSharedMemConfig")) : cuCtxSetSharedMemConfigNotFound;
    return func_ptr(config);
}

CUresult CUDAAPI cuModuleLoadDataExNotFound(CUmodule*, const void*, unsigned int, CUjit_option*, void**) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    using FuncPtr = CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleLoadDataEx")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleLoadDataEx")) : cuModuleLoadDataExNotFound;
    return func_ptr(module, image, numOptions, options, optionValues);
}

CUresult CUDAAPI cuModuleUnloadNotFound(CUmodule) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuModuleUnload(CUmodule hmod) {
    using FuncPtr = CUresult (*)(CUmodule);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleUnload")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleUnload")) : cuModuleUnloadNotFound;
    return func_ptr(hmod);
}

CUresult CUDAAPI cuModuleGetFunctionNotFound(CUfunction*, CUmodule, const char*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    using FuncPtr = CUresult (*)(CUfunction*, CUmodule, const char*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleGetFunction")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuModuleGetFunction")) : cuModuleGetFunctionNotFound;
    return func_ptr(hfunc, hmod, name);
}

CUresult CUDAAPI cuMemGetInfo_v2NotFound(size_t*, size_t*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
    using FuncPtr = CUresult (*)(size_t*, size_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetInfo_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetInfo_v2")) : cuMemGetInfo_v2NotFound;
    return func_ptr(free, total);
}

CUresult CUDAAPI cuMemAlloc_v2NotFound(CUdeviceptr*, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAlloc_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAlloc_v2")) : cuMemAlloc_v2NotFound;
    return func_ptr(dptr, bytesize);
}

CUresult CUDAAPI cuMemAllocPitch_v2NotFound(CUdeviceptr*, size_t*, size_t, size_t, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAllocPitch_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAllocPitch_v2")) : cuMemAllocPitch_v2NotFound;
    return func_ptr(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI cuMemFree_v2NotFound(CUdeviceptr) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    using FuncPtr = CUresult (*)(CUdeviceptr);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemFree_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemFree_v2")) : cuMemFree_v2NotFound;
    return func_ptr(dptr);
}

CUresult CUDAAPI cuMemGetAddressRange_v2NotFound(CUdeviceptr*, size_t*, CUdeviceptr) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAddressRange_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAddressRange_v2")) : cuMemGetAddressRange_v2NotFound;
    return func_ptr(pbase, psize, dptr);
}

CUresult CUDAAPI cuMemFreeHostNotFound(void*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemFreeHost(void* p) {
    using FuncPtr = CUresult (*)(void*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemFreeHost")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemFreeHost")) : cuMemFreeHostNotFound;
    return func_ptr(p);
}

CUresult CUDAAPI cuMemHostAllocNotFound(void**, size_t, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
    using FuncPtr = CUresult (*)(void**, size_t, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostAlloc")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostAlloc")) : cuMemHostAllocNotFound;
    return func_ptr(pp, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostGetDevicePointer_v2NotFound(CUdeviceptr*, void*, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, void*, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostGetDevicePointer_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostGetDevicePointer_v2")) : cuMemHostGetDevicePointer_v2NotFound;
    return func_ptr(pdptr, p, Flags);
}

CUresult CUDAAPI cuMemHostGetFlagsNotFound(unsigned int*, void*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
    using FuncPtr = CUresult (*)(unsigned int*, void*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostGetFlags")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostGetFlags")) : cuMemHostGetFlagsNotFound;
    return func_ptr(pFlags, p);
}

CUresult CUDAAPI cuMemAllocManagedNotFound(CUdeviceptr*, size_t, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, size_t, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAllocManaged")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAllocManaged")) : cuMemAllocManagedNotFound;
    return func_ptr(dptr, bytesize, flags);
}

CUresult CUDAAPI cuMemHostRegister_v2NotFound(void*, size_t, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) {
    using FuncPtr = CUresult (*)(void*, size_t, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostRegister_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostRegister_v2")) : cuMemHostRegister_v2NotFound;
    return func_ptr(p, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostUnregisterNotFound(void*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemHostUnregister(void* p) {
    using FuncPtr = CUresult (*)(void*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostUnregister")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemHostUnregister")) : cuMemHostUnregisterNotFound;
    return func_ptr(p);
}

CUresult CUDAAPI cuMemcpyNotFound(CUdeviceptr, CUdeviceptr, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    using FuncPtr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpy")) : cuMemcpyNotFound;
    return func_ptr(dst, src, ByteCount);
}

CUresult CUDAAPI cuMemcpyPeerNotFound(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    using FuncPtr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyPeer")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyPeer")) : cuMemcpyPeerNotFound;
    return func_ptr(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

CUresult CUDAAPI cuMemcpyAsyncNotFound(CUdeviceptr, CUdeviceptr, size_t, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyAsync")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyAsync")) : cuMemcpyAsyncNotFound;
    return func_ptr(dst, src, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyPeerAsyncNotFound(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyPeerAsync")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpyPeerAsync")) : cuMemcpyPeerAsyncNotFound;
    return func_ptr(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2DAsync_v2NotFound(const CUDA_MEMCPY2D*, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    using FuncPtr = CUresult (*)(const CUDA_MEMCPY2D*, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpy2DAsync_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemcpy2DAsync_v2")) : cuMemcpy2DAsync_v2NotFound;
    return func_ptr(pCopy, hStream);
}

CUresult CUDAAPI cuMemsetD8_v2NotFound(CUdeviceptr, unsigned char, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned char, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD8_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD8_v2")) : cuMemsetD8_v2NotFound;
    return func_ptr(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16_v2NotFound(CUdeviceptr, unsigned short, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned short, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD16_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD16_v2")) : cuMemsetD16_v2NotFound;
    return func_ptr(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32_v2NotFound(CUdeviceptr, unsigned int, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned int, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD32_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD32_v2")) : cuMemsetD32_v2NotFound;
    return func_ptr(dstDevice, ui, N);
}

CUresult CUDAAPI cuMemsetD8AsyncNotFound(CUdeviceptr, unsigned char, size_t, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD8Async")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD8Async")) : cuMemsetD8AsyncNotFound;
    return func_ptr(dstDevice, uc, N, hStream);
}

CUresult CUDAAPI cuMemsetD16AsyncNotFound(CUdeviceptr, unsigned short, size_t, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD16Async")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD16Async")) : cuMemsetD16AsyncNotFound;
    return func_ptr(dstDevice, us, N, hStream);
}

CUresult CUDAAPI cuMemsetD32AsyncNotFound(CUdeviceptr, unsigned int, size_t, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD32Async")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemsetD32Async")) : cuMemsetD32AsyncNotFound;
    return func_ptr(dstDevice, ui, N, hStream);
}

CUresult CUDAAPI cuMemAddressReserveNotFound(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    using FuncPtr = CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAddressReserve")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAddressReserve")) : cuMemAddressReserveNotFound;
    return func_ptr(ptr, size, alignment, addr, flags);
}

CUresult CUDAAPI cuMemAddressFreeNotFound(CUdeviceptr, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    using FuncPtr = CUresult (*)(CUdeviceptr, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAddressFree")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemAddressFree")) : cuMemAddressFreeNotFound;
    return func_ptr(ptr, size);
}

CUresult CUDAAPI cuMemCreateNotFound(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) {
    using FuncPtr = CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemCreate")) : cuMemCreateNotFound;
    return func_ptr(handle, size, prop, flags);
}

CUresult CUDAAPI cuMemReleaseNotFound(CUmemGenericAllocationHandle) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    using FuncPtr = CUresult (*)(CUmemGenericAllocationHandle);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRelease")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRelease")) : cuMemReleaseNotFound;
    return func_ptr(handle);
}

CUresult CUDAAPI cuMemMapNotFound(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    using FuncPtr = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemMap")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemMap")) : cuMemMapNotFound;
    return func_ptr(ptr, size, offset, handle, flags);
}

CUresult CUDAAPI cuMemUnmapNotFound(CUdeviceptr, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    using FuncPtr = CUresult (*)(CUdeviceptr, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemUnmap")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemUnmap")) : cuMemUnmapNotFound;
    return func_ptr(ptr, size);
}

CUresult CUDAAPI cuMemSetAccessNotFound(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    using FuncPtr = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemSetAccess")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemSetAccess")) : cuMemSetAccessNotFound;
    return func_ptr(ptr, size, desc, count);
}

CUresult CUDAAPI cuMemGetAccessNotFound(unsigned long long*, const CUmemLocation*, CUdeviceptr) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) {
    using FuncPtr = CUresult (*)(unsigned long long*, const CUmemLocation*, CUdeviceptr);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAccess")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAccess")) : cuMemGetAccessNotFound;
    return func_ptr(flags, location, ptr);
}

CUresult CUDAAPI cuMemExportToShareableHandleNotFound(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
    using FuncPtr = CUresult (*)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemExportToShareableHandle")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemExportToShareableHandle")) : cuMemExportToShareableHandleNotFound;
    return func_ptr(shareableHandle, handle, handleType, flags);
}

CUresult CUDAAPI cuMemImportFromShareableHandleNotFound(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) {
    using FuncPtr = CUresult (*)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemImportFromShareableHandle")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemImportFromShareableHandle")) : cuMemImportFromShareableHandleNotFound;
    return func_ptr(handle, osHandle, shHandleType);
}

CUresult CUDAAPI cuMemGetAllocationGranularityNotFound(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) {
    using FuncPtr = CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAllocationGranularity")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAllocationGranularity")) : cuMemGetAllocationGranularityNotFound;
    return func_ptr(granularity, prop, option);
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandleNotFound(CUmemAllocationProp*, CUmemGenericAllocationHandle) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    using FuncPtr = CUresult (*)(CUmemAllocationProp*, CUmemGenericAllocationHandle);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAllocationPropertiesFromHandle")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemGetAllocationPropertiesFromHandle")) : cuMemGetAllocationPropertiesFromHandleNotFound;
    return func_ptr(prop, handle);
}

CUresult CUDAAPI cuPointerGetAttributeNotFound(void*, CUpointer_attribute, CUdeviceptr) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    using FuncPtr = CUresult (*)(void*, CUpointer_attribute, CUdeviceptr);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuPointerGetAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuPointerGetAttribute")) : cuPointerGetAttributeNotFound;
    return func_ptr(data, attribute, ptr);
}

CUresult CUDAAPI cuMemRangeGetAttributeNotFound(void*, size_t, CUmem_range_attribute, CUdeviceptr, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
    using FuncPtr = CUresult (*)(void*, size_t, CUmem_range_attribute, CUdeviceptr, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRangeGetAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRangeGetAttribute")) : cuMemRangeGetAttributeNotFound;
    return func_ptr(data, dataSize, attribute, devPtr, count);
}

CUresult CUDAAPI cuMemRangeGetAttributesNotFound(void**, size_t*, CUmem_range_attribute*, size_t, CUdeviceptr, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
    using FuncPtr = CUresult (*)(void**, size_t*, CUmem_range_attribute*, size_t, CUdeviceptr, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRangeGetAttributes")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuMemRangeGetAttributes")) : cuMemRangeGetAttributesNotFound;
    return func_ptr(data, dataSizes, attributes, numAttributes, devPtr, count);
}

CUresult CUDAAPI cuPointerGetAttributesNotFound(unsigned int, CUpointer_attribute*, void**, CUdeviceptr) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) {
    using FuncPtr = CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuPointerGetAttributes")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuPointerGetAttributes")) : cuPointerGetAttributesNotFound;
    return func_ptr(numAttributes, attributes, data, ptr);
}

CUresult CUDAAPI cuStreamCreateNotFound(CUstream*, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    using FuncPtr = CUresult (*)(CUstream*, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamCreate")) : cuStreamCreateNotFound;
    return func_ptr(phStream, Flags);
}

CUresult CUDAAPI cuStreamCreateWithPriorityNotFound(CUstream*, unsigned int, int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
    using FuncPtr = CUresult (*)(CUstream*, unsigned int, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamCreateWithPriority")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamCreateWithPriority")) : cuStreamCreateWithPriorityNotFound;
    return func_ptr(phStream, flags, priority);
}

CUresult CUDAAPI cuStreamGetPriorityNotFound(CUstream, int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamGetPriority(CUstream hStream, int* priority) {
    using FuncPtr = CUresult (*)(CUstream, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetPriority")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetPriority")) : cuStreamGetPriorityNotFound;
    return func_ptr(hStream, priority);
}

CUresult CUDAAPI cuStreamGetFlagsNotFound(CUstream, unsigned int*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
    using FuncPtr = CUresult (*)(CUstream, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetFlags")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetFlags")) : cuStreamGetFlagsNotFound;
    return func_ptr(hStream, flags);
}

CUresult CUDAAPI cuStreamGetCtxNotFound(CUstream, CUcontext*) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) {
    using FuncPtr = CUresult (*)(CUstream, CUcontext*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetCtx")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamGetCtx")) : cuStreamGetCtxNotFound;
    return func_ptr(hStream, pctx);
}

CUresult CUDAAPI cuStreamWaitEventNotFound(CUstream, CUevent, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    using FuncPtr = CUresult (*)(CUstream, CUevent, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamWaitEvent")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamWaitEvent")) : cuStreamWaitEventNotFound;
    return func_ptr(hStream, hEvent, Flags);
}

CUresult CUDAAPI cuStreamQueryNotFound(CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamQuery(CUstream hStream) {
    using FuncPtr = CUresult (*)(CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamQuery")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamQuery")) : cuStreamQueryNotFound;
    return func_ptr(hStream);
}

CUresult CUDAAPI cuStreamSynchronizeNotFound(CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    using FuncPtr = CUresult (*)(CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamSynchronize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamSynchronize")) : cuStreamSynchronizeNotFound;
    return func_ptr(hStream);
}

CUresult CUDAAPI cuStreamDestroy_v2NotFound(CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    using FuncPtr = CUresult (*)(CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamDestroy_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuStreamDestroy_v2")) : cuStreamDestroy_v2NotFound;
    return func_ptr(hStream);
}

CUresult CUDAAPI cuEventCreateNotFound(CUevent*, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    using FuncPtr = CUresult (*)(CUevent*, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventCreate")) : cuEventCreateNotFound;
    return func_ptr(phEvent, Flags);
}

CUresult CUDAAPI cuEventRecordNotFound(CUevent, CUstream) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    using FuncPtr = CUresult (*)(CUevent, CUstream);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventRecord")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventRecord")) : cuEventRecordNotFound;
    return func_ptr(hEvent, hStream);
}

CUresult CUDAAPI cuEventQueryNotFound(CUevent) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventQuery(CUevent hEvent) {
    using FuncPtr = CUresult (*)(CUevent);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventQuery")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventQuery")) : cuEventQueryNotFound;
    return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventSynchronizeNotFound(CUevent) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    using FuncPtr = CUresult (*)(CUevent);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventSynchronize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventSynchronize")) : cuEventSynchronizeNotFound;
    return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventDestroy_v2NotFound(CUevent) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    using FuncPtr = CUresult (*)(CUevent);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventDestroy_v2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventDestroy_v2")) : cuEventDestroy_v2NotFound;
    return func_ptr(hEvent);
}

CUresult CUDAAPI cuEventElapsedTimeNotFound(float*, CUevent, CUevent) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    using FuncPtr = CUresult (*)(float*, CUevent, CUevent);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventElapsedTime")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuEventElapsedTime")) : cuEventElapsedTimeNotFound;
    return func_ptr(pMilliseconds, hStart, hEnd);
}

CUresult CUDAAPI cuFuncGetAttributeNotFound(int*, CUfunction_attribute, CUfunction) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
    using FuncPtr = CUresult (*)(int*, CUfunction_attribute, CUfunction);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncGetAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncGetAttribute")) : cuFuncGetAttributeNotFound;
    return func_ptr(pi, attrib, hfunc);
}

CUresult CUDAAPI cuFuncSetAttributeNotFound(CUfunction, CUfunction_attribute, int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    using FuncPtr = CUresult (*)(CUfunction, CUfunction_attribute, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetAttribute")) : cuFuncSetAttributeNotFound;
    return func_ptr(hfunc, attrib, value);
}

CUresult CUDAAPI cuFuncSetCacheConfigNotFound(CUfunction, CUfunc_cache) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    using FuncPtr = CUresult (*)(CUfunction, CUfunc_cache);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetCacheConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetCacheConfig")) : cuFuncSetCacheConfigNotFound;
    return func_ptr(hfunc, config);
}

CUresult CUDAAPI cuLaunchKernelNotFound(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    using FuncPtr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuLaunchKernel")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuLaunchKernel")) : cuLaunchKernelNotFound;
    return func_ptr(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchCooperativeKernelNotFound(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) {
    using FuncPtr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuLaunchCooperativeKernel")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuLaunchCooperativeKernel")) : cuLaunchCooperativeKernelNotFound;
    return func_ptr(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult CUDAAPI cuFuncSetSharedMemConfigNotFound(CUfunction, CUsharedconfig) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    using FuncPtr = CUresult (*)(CUfunction, CUsharedconfig);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetSharedMemConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuFuncSetSharedMemConfig")) : cuFuncSetSharedMemConfigNotFound;
    return func_ptr(hfunc, config);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorNotFound(int*, CUfunction, int, size_t) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    using FuncPtr = CUresult (*)(int*, CUfunction, int, size_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxActiveBlocksPerMultiprocessor")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxActiveBlocksPerMultiprocessor")) : cuOccupancyMaxActiveBlocksPerMultiprocessorNotFound;
    return func_ptr(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNotFound(int*, CUfunction, int, size_t, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    using FuncPtr = CUresult (*)(int*, CUfunction, int, size_t, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")) : cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsNotFound;
    return func_ptr(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeNotFound(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
    using FuncPtr = CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxPotentialBlockSize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxPotentialBlockSize")) : cuOccupancyMaxPotentialBlockSizeNotFound;
    return func_ptr(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlagsNotFound(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
    using FuncPtr = CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxPotentialBlockSizeWithFlags")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuOccupancyMaxPotentialBlockSizeWithFlags")) : cuOccupancyMaxPotentialBlockSizeWithFlagsNotFound;
    return func_ptr(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
}

CUresult CUDAAPI cuDeviceCanAccessPeerNotFound(int*, CUdevice, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) {
    using FuncPtr = CUresult (*)(int*, CUdevice, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceCanAccessPeer")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceCanAccessPeer")) : cuDeviceCanAccessPeerNotFound;
    return func_ptr(canAccessPeer, dev, peerDev);
}

CUresult CUDAAPI cuCtxEnablePeerAccessNotFound(CUcontext, unsigned int) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    using FuncPtr = CUresult (*)(CUcontext, unsigned int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxEnablePeerAccess")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxEnablePeerAccess")) : cuCtxEnablePeerAccessNotFound;
    return func_ptr(peerContext, Flags);
}

CUresult CUDAAPI cuCtxDisablePeerAccessNotFound(CUcontext) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    using FuncPtr = CUresult (*)(CUcontext);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxDisablePeerAccess")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuCtxDisablePeerAccess")) : cuCtxDisablePeerAccessNotFound;
    return func_ptr(peerContext);
}

CUresult CUDAAPI cuDeviceGetP2PAttributeNotFound(int*, CUdevice_P2PAttribute, CUdevice, CUdevice) {
    return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
    using FuncPtr = CUresult (*)(int*, CUdevice_P2PAttribute, CUdevice, CUdevice);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetP2PAttribute")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuDeviceGetP2PAttribute")) : cuDeviceGetP2PAttributeNotFound;
    return func_ptr(value, attrib, srcDevice, dstDevice);
}
