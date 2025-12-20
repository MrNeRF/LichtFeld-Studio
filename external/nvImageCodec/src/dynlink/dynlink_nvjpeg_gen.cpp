#include <nvjpeg.h>

void* NvjpegLoadSymbol(const char* name);

#define LOAD_SYMBOL_FUNC Nvjpeg##LoadSymbol

nvjpegStatus_t NVJPEGAPI nvjpegGetPropertyNotFound(libraryPropertyType, int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegGetProperty(libraryPropertyType type, int* value) {
    using FuncPtr = nvjpegStatus_t (*)(libraryPropertyType, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetProperty")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetProperty")) : nvjpegGetPropertyNotFound;
    return func_ptr(type, value);
}

nvjpegStatus_t NVJPEGAPI nvjpegCreateNotFound(nvjpegBackend_t, nvjpegDevAllocator_t*, nvjpegHandle_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t* dev_allocator, nvjpegHandle_t* handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBackend_t, nvjpegDevAllocator_t*, nvjpegHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreate")) : nvjpegCreateNotFound;
    return func_ptr(backend, dev_allocator, handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegCreateSimpleNotFound(nvjpegHandle_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t* handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateSimple")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateSimple")) : nvjpegCreateSimpleNotFound;
    return func_ptr(handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegCreateExNotFound(nvjpegBackend_t, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*, unsigned int, nvjpegHandle_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t backend, nvjpegDevAllocator_t* dev_allocator, nvjpegPinnedAllocator_t* pinned_allocator, unsigned int flags, nvjpegHandle_t* handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBackend_t, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*, unsigned int, nvjpegHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateEx")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateEx")) : nvjpegCreateExNotFound;
    return func_ptr(backend, dev_allocator, pinned_allocator, flags, handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegCreateExV2NotFound(nvjpegBackend_t, nvjpegDevAllocatorV2_t*, nvjpegPinnedAllocatorV2_t*, unsigned int, nvjpegHandle_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegCreateExV2(nvjpegBackend_t backend, nvjpegDevAllocatorV2_t* dev_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, unsigned int flags, nvjpegHandle_t* handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBackend_t, nvjpegDevAllocatorV2_t*, nvjpegPinnedAllocatorV2_t*, unsigned int, nvjpegHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateExV2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegCreateExV2")) : nvjpegCreateExV2NotFound;
    return func_ptr(backend, dev_allocator, pinned_allocator, flags, handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegDestroyNotFound(nvjpegHandle_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDestroy")) : nvjpegDestroyNotFound;
    return func_ptr(handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegSetDeviceMemoryPaddingNotFound(size_t, nvjpegHandle_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegSetDeviceMemoryPadding(size_t padding, nvjpegHandle_t handle) {
    using FuncPtr = nvjpegStatus_t (*)(size_t, nvjpegHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegSetDeviceMemoryPadding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegSetDeviceMemoryPadding")) : nvjpegSetDeviceMemoryPaddingNotFound;
    return func_ptr(padding, handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegSetPinnedMemoryPaddingNotFound(size_t, nvjpegHandle_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegSetPinnedMemoryPadding(size_t padding, nvjpegHandle_t handle) {
    using FuncPtr = nvjpegStatus_t (*)(size_t, nvjpegHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegSetPinnedMemoryPadding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegSetPinnedMemoryPadding")) : nvjpegSetPinnedMemoryPaddingNotFound;
    return func_ptr(padding, handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegGetHardwareDecoderInfoNotFound(nvjpegHandle_t, unsigned int*, unsigned int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegGetHardwareDecoderInfo(nvjpegHandle_t handle, unsigned int* num_engines, unsigned int* num_cores_per_engine) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, unsigned int*, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetHardwareDecoderInfo")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetHardwareDecoderInfo")) : nvjpegGetHardwareDecoderInfoNotFound;
    return func_ptr(handle, num_engines, num_cores_per_engine);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStateCreateNotFound(nvjpegHandle_t, nvjpegJpegState_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t* jpeg_handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStateCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStateCreate")) : nvjpegJpegStateCreateNotFound;
    return func_ptr(handle, jpeg_handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStateDestroyNotFound(nvjpegJpegState_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t jpeg_handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegState_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStateDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStateDestroy")) : nvjpegJpegStateDestroyNotFound;
    return func_ptr(jpeg_handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegGetImageInfoNotFound(nvjpegHandle_t, const unsigned char*, size_t, int*, nvjpegChromaSubsampling_t*, int*, int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t handle, const unsigned char* data, size_t length, int* nComponents, nvjpegChromaSubsampling_t* subsampling, int* widths, int* heights) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char*, size_t, int*, nvjpegChromaSubsampling_t*, int*, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetImageInfo")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegGetImageInfo")) : nvjpegGetImageInfoNotFound;
    return func_ptr(handle, data, length, nComponents, subsampling, widths, heights);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedInitializeNotFound(nvjpegHandle_t, nvjpegJpegState_t, int, int, nvjpegOutputFormat_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatchedInitialize(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, int batch_size, int max_cpu_threads, nvjpegOutputFormat_t output_format) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, int, int, nvjpegOutputFormat_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedInitialize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedInitialize")) : nvjpegDecodeBatchedInitializeNotFound;
    return func_ptr(handle, jpeg_handle, batch_size, max_cpu_threads, output_format);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedNotFound(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char* const*, const size_t*, nvjpegImage_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatched(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, const unsigned char* const* data, const size_t* lengths, nvjpegImage_t* destinations, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char* const*, const size_t*, nvjpegImage_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatched")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatched")) : nvjpegDecodeBatchedNotFound;
    return func_ptr(handle, jpeg_handle, data, lengths, destinations, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedPreAllocateNotFound(nvjpegHandle_t, nvjpegJpegState_t, int, int, int, nvjpegChromaSubsampling_t, nvjpegOutputFormat_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatchedPreAllocate(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, int batch_size, int width, int height, nvjpegChromaSubsampling_t chroma_subsampling, nvjpegOutputFormat_t output_format) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, int, int, int, nvjpegChromaSubsampling_t, nvjpegOutputFormat_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedPreAllocate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedPreAllocate")) : nvjpegDecodeBatchedPreAllocateNotFound;
    return func_ptr(handle, jpeg_handle, batch_size, width, height, chroma_subsampling, output_format);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderStateCreateNotFound(nvjpegHandle_t, nvjpegEncoderState_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t handle, nvjpegEncoderState_t* encoder_state, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderStateCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderStateCreate")) : nvjpegEncoderStateCreateNotFound;
    return func_ptr(handle, encoder_state, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderStateDestroyNotFound(nvjpegEncoderState_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t encoder_state) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderState_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderStateDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderStateDestroy")) : nvjpegEncoderStateDestroyNotFound;
    return func_ptr(encoder_state);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsCreateNotFound(nvjpegHandle_t, nvjpegEncoderParams_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t handle, nvjpegEncoderParams_t* encoder_params, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderParams_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsCreate")) : nvjpegEncoderParamsCreateNotFound;
    return func_ptr(handle, encoder_params, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsDestroyNotFound(nvjpegEncoderParams_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t encoder_params) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderParams_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsDestroy")) : nvjpegEncoderParamsDestroyNotFound;
    return func_ptr(encoder_params);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetQualityNotFound(nvjpegEncoderParams_t, const int, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t encoder_params, const int quality, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const int, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetQuality")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetQuality")) : nvjpegEncoderParamsSetQualityNotFound;
    return func_ptr(encoder_params, quality, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetEncodingNotFound(nvjpegEncoderParams_t, nvjpegJpegEncoding_t, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsSetEncoding(nvjpegEncoderParams_t encoder_params, nvjpegJpegEncoding_t etype, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, nvjpegJpegEncoding_t, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetEncoding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetEncoding")) : nvjpegEncoderParamsSetEncodingNotFound;
    return func_ptr(encoder_params, etype, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetOptimizedHuffmanNotFound(nvjpegEncoderParams_t, const int, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t encoder_params, const int optimized, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const int, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetOptimizedHuffman")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetOptimizedHuffman")) : nvjpegEncoderParamsSetOptimizedHuffmanNotFound;
    return func_ptr(encoder_params, optimized, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncoderParamsSetSamplingFactorsNotFound(nvjpegEncoderParams_t, const nvjpegChromaSubsampling_t, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams_t encoder_params, const nvjpegChromaSubsampling_t chroma_subsampling, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const nvjpegChromaSubsampling_t, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetSamplingFactors")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncoderParamsSetSamplingFactors")) : nvjpegEncoderParamsSetSamplingFactorsNotFound;
    return func_ptr(encoder_params, chroma_subsampling, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncodeYUVNotFound(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t*, nvjpegChromaSubsampling_t, int, int, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncodeYUV(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state, const nvjpegEncoderParams_t encoder_params, const nvjpegImage_t* source, nvjpegChromaSubsampling_t chroma_subsampling, int image_width, int image_height, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t*, nvjpegChromaSubsampling_t, int, int, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeYUV")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeYUV")) : nvjpegEncodeYUVNotFound;
    return func_ptr(handle, encoder_state, encoder_params, source, chroma_subsampling, image_width, image_height, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncodeImageNotFound(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t*, nvjpegInputFormat_t, int, int, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncodeImage(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state, const nvjpegEncoderParams_t encoder_params, const nvjpegImage_t* source, nvjpegInputFormat_t input_format, int image_width, int image_height, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t, const nvjpegImage_t*, nvjpegInputFormat_t, int, int, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeImage")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeImage")) : nvjpegEncodeImageNotFound;
    return func_ptr(handle, encoder_state, encoder_params, source, input_format, image_width, image_height, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegEncodeRetrieveBitstreamNotFound(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char*, size_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state, unsigned char* data, size_t* length, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char*, size_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeRetrieveBitstream")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegEncodeRetrieveBitstream")) : nvjpegEncodeRetrieveBitstreamNotFound;
    return func_ptr(handle, encoder_state, data, length, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedCreateNotFound(nvjpegHandle_t, nvjpegPinnedAllocator_t*, nvjpegBufferPinned_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferPinnedCreate(nvjpegHandle_t handle, nvjpegPinnedAllocator_t* pinned_allocator, nvjpegBufferPinned_t* buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegPinnedAllocator_t*, nvjpegBufferPinned_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedCreate")) : nvjpegBufferPinnedCreateNotFound;
    return func_ptr(handle, pinned_allocator, buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedCreateV2NotFound(nvjpegHandle_t, nvjpegPinnedAllocatorV2_t*, nvjpegBufferPinned_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferPinnedCreateV2(nvjpegHandle_t handle, nvjpegPinnedAllocatorV2_t* pinned_allocator, nvjpegBufferPinned_t* buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegPinnedAllocatorV2_t*, nvjpegBufferPinned_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedCreateV2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedCreateV2")) : nvjpegBufferPinnedCreateV2NotFound;
    return func_ptr(handle, pinned_allocator, buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedResizeNotFound(nvjpegBufferPinned_t, size_t, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferPinnedResize(nvjpegBufferPinned_t buffer, size_t size, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBufferPinned_t, size_t, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedResize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedResize")) : nvjpegBufferPinnedResizeNotFound;
    return func_ptr(buffer, size, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferPinnedDestroyNotFound(nvjpegBufferPinned_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferPinnedDestroy(nvjpegBufferPinned_t buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBufferPinned_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferPinnedDestroy")) : nvjpegBufferPinnedDestroyNotFound;
    return func_ptr(buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceCreateNotFound(nvjpegHandle_t, nvjpegDevAllocator_t*, nvjpegBufferDevice_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferDeviceCreate(nvjpegHandle_t handle, nvjpegDevAllocator_t* device_allocator, nvjpegBufferDevice_t* buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegDevAllocator_t*, nvjpegBufferDevice_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceCreate")) : nvjpegBufferDeviceCreateNotFound;
    return func_ptr(handle, device_allocator, buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceCreateV2NotFound(nvjpegHandle_t, nvjpegDevAllocatorV2_t*, nvjpegBufferDevice_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferDeviceCreateV2(nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegBufferDevice_t* buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegDevAllocatorV2_t*, nvjpegBufferDevice_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceCreateV2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceCreateV2")) : nvjpegBufferDeviceCreateV2NotFound;
    return func_ptr(handle, device_allocator, buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceResizeNotFound(nvjpegBufferDevice_t, size_t, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferDeviceResize(nvjpegBufferDevice_t buffer, size_t size, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBufferDevice_t, size_t, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceResize")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceResize")) : nvjpegBufferDeviceResizeNotFound;
    return func_ptr(buffer, size, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegBufferDeviceDestroyNotFound(nvjpegBufferDevice_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegBufferDeviceDestroy(nvjpegBufferDevice_t buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegBufferDevice_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegBufferDeviceDestroy")) : nvjpegBufferDeviceDestroyNotFound;
    return func_ptr(buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegStateAttachPinnedBufferNotFound(nvjpegJpegState_t, nvjpegBufferPinned_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegStateAttachPinnedBuffer(nvjpegJpegState_t decoder_state, nvjpegBufferPinned_t pinned_buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegState_t, nvjpegBufferPinned_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegStateAttachPinnedBuffer")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegStateAttachPinnedBuffer")) : nvjpegStateAttachPinnedBufferNotFound;
    return func_ptr(decoder_state, pinned_buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegStateAttachDeviceBufferNotFound(nvjpegJpegState_t, nvjpegBufferDevice_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegStateAttachDeviceBuffer(nvjpegJpegState_t decoder_state, nvjpegBufferDevice_t device_buffer) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegState_t, nvjpegBufferDevice_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegStateAttachDeviceBuffer")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegStateAttachDeviceBuffer")) : nvjpegStateAttachDeviceBufferNotFound;
    return func_ptr(decoder_state, device_buffer);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamCreateNotFound(nvjpegHandle_t, nvjpegJpegStream_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamCreate(nvjpegHandle_t handle, nvjpegJpegStream_t* jpeg_stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamCreate")) : nvjpegJpegStreamCreateNotFound;
    return func_ptr(handle, jpeg_stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamDestroyNotFound(nvjpegJpegStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamDestroy(nvjpegJpegStream_t jpeg_stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamDestroy")) : nvjpegJpegStreamDestroyNotFound;
    return func_ptr(jpeg_stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamParseNotFound(nvjpegHandle_t, const unsigned char*, size_t, int, int, nvjpegJpegStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamParse(nvjpegHandle_t handle, const unsigned char* data, size_t length, int save_metadata, int save_stream, nvjpegJpegStream_t jpeg_stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char*, size_t, int, int, nvjpegJpegStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamParse")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamParse")) : nvjpegJpegStreamParseNotFound;
    return func_ptr(handle, data, length, save_metadata, save_stream, jpeg_stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamParseHeaderNotFound(nvjpegHandle_t, const unsigned char*, size_t, nvjpegJpegStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamParseHeader(nvjpegHandle_t handle, const unsigned char* data, size_t length, nvjpegJpegStream_t jpeg_stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char*, size_t, nvjpegJpegStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamParseHeader")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamParseHeader")) : nvjpegJpegStreamParseHeaderNotFound;
    return func_ptr(handle, data, length, jpeg_stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetJpegEncodingNotFound(nvjpegJpegStream_t, nvjpegJpegEncoding_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetJpegEncoding(nvjpegJpegStream_t jpeg_stream, nvjpegJpegEncoding_t* jpeg_encoding) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, nvjpegJpegEncoding_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetJpegEncoding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetJpegEncoding")) : nvjpegJpegStreamGetJpegEncodingNotFound;
    return func_ptr(jpeg_stream, jpeg_encoding);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetFrameDimensionsNotFound(nvjpegJpegStream_t, unsigned int*, unsigned int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetFrameDimensions(nvjpegJpegStream_t jpeg_stream, unsigned int* width, unsigned int* height) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int*, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetFrameDimensions")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetFrameDimensions")) : nvjpegJpegStreamGetFrameDimensionsNotFound;
    return func_ptr(jpeg_stream, width, height);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetComponentsNumNotFound(nvjpegJpegStream_t, unsigned int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetComponentsNum(nvjpegJpegStream_t jpeg_stream, unsigned int* components_num) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetComponentsNum")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetComponentsNum")) : nvjpegJpegStreamGetComponentsNumNotFound;
    return func_ptr(jpeg_stream, components_num);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetComponentDimensionsNotFound(nvjpegJpegStream_t, unsigned int, unsigned int*, unsigned int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetComponentDimensions(nvjpegJpegStream_t jpeg_stream, unsigned int component, unsigned int* width, unsigned int* height) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int, unsigned int*, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetComponentDimensions")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetComponentDimensions")) : nvjpegJpegStreamGetComponentDimensionsNotFound;
    return func_ptr(jpeg_stream, component, width, height);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetExifOrientationNotFound(nvjpegJpegStream_t, nvjpegExifOrientation_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetExifOrientation(nvjpegJpegStream_t jpeg_stream, nvjpegExifOrientation_t* orientation_flag) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, nvjpegExifOrientation_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetExifOrientation")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetExifOrientation")) : nvjpegJpegStreamGetExifOrientationNotFound;
    return func_ptr(jpeg_stream, orientation_flag);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetSamplePrecisionNotFound(nvjpegJpegStream_t, unsigned int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetSamplePrecision(nvjpegJpegStream_t jpeg_stream, unsigned int* precision) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetSamplePrecision")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetSamplePrecision")) : nvjpegJpegStreamGetSamplePrecisionNotFound;
    return func_ptr(jpeg_stream, precision);
}

nvjpegStatus_t NVJPEGAPI nvjpegJpegStreamGetChromaSubsamplingNotFound(nvjpegJpegStream_t, nvjpegChromaSubsampling_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegJpegStreamGetChromaSubsampling(nvjpegJpegStream_t jpeg_stream, nvjpegChromaSubsampling_t* chroma_subsampling) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegStream_t, nvjpegChromaSubsampling_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetChromaSubsampling")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegJpegStreamGetChromaSubsampling")) : nvjpegJpegStreamGetChromaSubsamplingNotFound;
    return func_ptr(jpeg_stream, chroma_subsampling);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsCreateNotFound(nvjpegHandle_t, nvjpegDecodeParams_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsCreate(nvjpegHandle_t handle, nvjpegDecodeParams_t* decode_params) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegDecodeParams_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsCreate")) : nvjpegDecodeParamsCreateNotFound;
    return func_ptr(handle, decode_params);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsDestroyNotFound(nvjpegDecodeParams_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsDestroy(nvjpegDecodeParams_t decode_params) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegDecodeParams_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsDestroy")) : nvjpegDecodeParamsDestroyNotFound;
    return func_ptr(decode_params);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetOutputFormatNotFound(nvjpegDecodeParams_t, nvjpegOutputFormat_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsSetOutputFormat(nvjpegDecodeParams_t decode_params, nvjpegOutputFormat_t output_format) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, nvjpegOutputFormat_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetOutputFormat")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetOutputFormat")) : nvjpegDecodeParamsSetOutputFormatNotFound;
    return func_ptr(decode_params, output_format);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetROINotFound(nvjpegDecodeParams_t, int, int, int, int) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsSetROI(nvjpegDecodeParams_t decode_params, int offset_x, int offset_y, int roi_width, int roi_height) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, int, int, int, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetROI")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetROI")) : nvjpegDecodeParamsSetROINotFound;
    return func_ptr(decode_params, offset_x, offset_y, roi_width, roi_height);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetAllowCMYKNotFound(nvjpegDecodeParams_t, int) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsSetAllowCMYK(nvjpegDecodeParams_t decode_params, int allow_cmyk) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, int);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetAllowCMYK")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetAllowCMYK")) : nvjpegDecodeParamsSetAllowCMYKNotFound;
    return func_ptr(decode_params, allow_cmyk);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeParamsSetExifOrientationNotFound(nvjpegDecodeParams_t, nvjpegExifOrientation_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeParamsSetExifOrientation(nvjpegDecodeParams_t decode_params, nvjpegExifOrientation_t orientation) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, nvjpegExifOrientation_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetExifOrientation")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeParamsSetExifOrientation")) : nvjpegDecodeParamsSetExifOrientationNotFound;
    return func_ptr(decode_params, orientation);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecoderCreateNotFound(nvjpegHandle_t, nvjpegBackend_t, nvjpegJpegDecoder_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecoderCreate(nvjpegHandle_t nvjpeg_handle, nvjpegBackend_t implementation, nvjpegJpegDecoder_t* decoder_handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegBackend_t, nvjpegJpegDecoder_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderCreate")) : nvjpegDecoderCreateNotFound;
    return func_ptr(nvjpeg_handle, implementation, decoder_handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecoderDestroyNotFound(nvjpegJpegDecoder_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecoderDestroy(nvjpegJpegDecoder_t decoder_handle) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegDecoder_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderDestroy")) : nvjpegDecoderDestroyNotFound;
    return func_ptr(decoder_handle);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecoderJpegSupportedNotFound(nvjpegJpegDecoder_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecoderJpegSupported(nvjpegJpegDecoder_t decoder_handle, nvjpegJpegStream_t jpeg_stream, nvjpegDecodeParams_t decode_params, int* is_supported) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegJpegDecoder_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderJpegSupported")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderJpegSupported")) : nvjpegDecoderJpegSupportedNotFound;
    return func_ptr(decoder_handle, jpeg_stream, decode_params, is_supported);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedSupportedNotFound(nvjpegHandle_t, nvjpegJpegStream_t, int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatchedSupported(nvjpegHandle_t handle, nvjpegJpegStream_t jpeg_stream, int* is_supported) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedSupported")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedSupported")) : nvjpegDecodeBatchedSupportedNotFound;
    return func_ptr(handle, jpeg_stream, is_supported);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedSupportedExNotFound(nvjpegHandle_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatchedSupportedEx(nvjpegHandle_t handle, nvjpegJpegStream_t jpeg_stream, nvjpegDecodeParams_t decode_params, int* is_supported) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedSupportedEx")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedSupportedEx")) : nvjpegDecodeBatchedSupportedExNotFound;
    return func_ptr(handle, jpeg_stream, decode_params, is_supported);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecoderStateCreateNotFound(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t*) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecoderStateCreate(nvjpegHandle_t nvjpeg_handle, nvjpegJpegDecoder_t decoder_handle, nvjpegJpegState_t* decoder_state) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderStateCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecoderStateCreate")) : nvjpegDecoderStateCreateNotFound;
    return func_ptr(nvjpeg_handle, decoder_handle, decoder_state);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegHostNotFound(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegDecodeParams_t, nvjpegJpegStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeJpegHost(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder, nvjpegJpegState_t decoder_state, nvjpegDecodeParams_t decode_params, nvjpegJpegStream_t jpeg_stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegDecodeParams_t, nvjpegJpegStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegHost")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegHost")) : nvjpegDecodeJpegHostNotFound;
    return func_ptr(handle, decoder, decoder_state, decode_params, jpeg_stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegTransferToDeviceNotFound(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeJpegTransferToDevice(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder, nvjpegJpegState_t decoder_state, nvjpegJpegStream_t jpeg_stream, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegTransferToDevice")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegTransferToDevice")) : nvjpegDecodeJpegTransferToDeviceNotFound;
    return func_ptr(handle, decoder, decoder_state, jpeg_stream, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeJpegDeviceNotFound(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegImage_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeJpegDevice(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder, nvjpegJpegState_t decoder_state, nvjpegImage_t* destination, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegImage_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegDevice")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeJpegDevice")) : nvjpegDecodeJpegDeviceNotFound;
    return func_ptr(handle, decoder, decoder_state, destination, stream);
}

nvjpegStatus_t NVJPEGAPI nvjpegDecodeBatchedExNotFound(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char* const*, const size_t*, nvjpegImage_t*, nvjpegDecodeParams_t*, cudaStream_t) {
    return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpegStatus_t nvjpegDecodeBatchedEx(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle, const unsigned char* const* data, const size_t* lengths, nvjpegImage_t* destinations, nvjpegDecodeParams_t* decode_params, cudaStream_t stream) {
    using FuncPtr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char* const*, const size_t*, nvjpegImage_t*, nvjpegDecodeParams_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedEx")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpegDecodeBatchedEx")) : nvjpegDecodeBatchedExNotFound;
    return func_ptr(handle, jpeg_handle, data, lengths, destinations, decode_params, stream);
}
