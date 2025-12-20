#include <nvjpeg2k.h>

void* Nvjpeg2kLoadSymbol(const char* name);

#define LOAD_SYMBOL_FUNC Nvjpeg2k##LoadSymbol

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kGetPropertyNotFound(libraryPropertyType, int*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kGetProperty(libraryPropertyType type, int* value) {
    using FuncPtr = nvjpeg2kStatus_t (*)(libraryPropertyType, int*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kGetProperty")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kGetProperty")) : nvjpeg2kGetPropertyNotFound;
    return func_ptr(type, value);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kCreateSimpleNotFound(nvjpeg2kHandle_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kCreateSimple(nvjpeg2kHandle_t* handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kCreateSimple")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kCreateSimple")) : nvjpeg2kCreateSimpleNotFound;
    return func_ptr(handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kCreateV2NotFound(nvjpeg2kBackend_t, nvjpeg2kDeviceAllocatorV2_t*, nvjpeg2kPinnedAllocatorV2_t*, nvjpeg2kHandle_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kCreateV2(nvjpeg2kBackend_t backend, nvjpeg2kDeviceAllocatorV2_t* dev_allocator, nvjpeg2kPinnedAllocatorV2_t* pinned_allocator, nvjpeg2kHandle_t* handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kBackend_t, nvjpeg2kDeviceAllocatorV2_t*, nvjpeg2kPinnedAllocatorV2_t*, nvjpeg2kHandle_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kCreateV2")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kCreateV2")) : nvjpeg2kCreateV2NotFound;
    return func_ptr(backend, dev_allocator, pinned_allocator, handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDestroyNotFound(nvjpeg2kHandle_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDestroy(nvjpeg2kHandle_t handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDestroy")) : nvjpeg2kDestroyNotFound;
    return func_ptr(handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kSetDeviceMemoryPaddingNotFound(size_t, nvjpeg2kHandle_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kSetDeviceMemoryPadding(size_t padding, nvjpeg2kHandle_t handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(size_t, nvjpeg2kHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kSetDeviceMemoryPadding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kSetDeviceMemoryPadding")) : nvjpeg2kSetDeviceMemoryPaddingNotFound;
    return func_ptr(padding, handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kSetPinnedMemoryPaddingNotFound(size_t, nvjpeg2kHandle_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kSetPinnedMemoryPadding(size_t padding, nvjpeg2kHandle_t handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(size_t, nvjpeg2kHandle_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kSetPinnedMemoryPadding")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kSetPinnedMemoryPadding")) : nvjpeg2kSetPinnedMemoryPaddingNotFound;
    return func_ptr(padding, handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeStateCreateNotFound(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeStateCreate(nvjpeg2kHandle_t handle, nvjpeg2kDecodeState_t* decode_state) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeStateCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeStateCreate")) : nvjpeg2kDecodeStateCreateNotFound;
    return func_ptr(handle, decode_state);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeStateDestroyNotFound(nvjpeg2kDecodeState_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeStateDestroy(nvjpeg2kDecodeState_t decode_state) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeState_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeStateDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeStateDestroy")) : nvjpeg2kDecodeStateDestroyNotFound;
    return func_ptr(decode_state);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kStreamCreateNotFound(nvjpeg2kStream_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kStreamCreate(nvjpeg2kStream_t* stream_handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kStream_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamCreate")) : nvjpeg2kStreamCreateNotFound;
    return func_ptr(stream_handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kStreamDestroyNotFound(nvjpeg2kStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kStreamDestroy(nvjpeg2kStream_t stream_handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamDestroy")) : nvjpeg2kStreamDestroyNotFound;
    return func_ptr(stream_handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kStreamParseNotFound(nvjpeg2kHandle_t, const unsigned char*, size_t, int, int, nvjpeg2kStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kStreamParse(nvjpeg2kHandle_t handle, const unsigned char* data, size_t length, int save_metadata, int save_stream, nvjpeg2kStream_t stream_handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t, const unsigned char*, size_t, int, int, nvjpeg2kStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamParse")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamParse")) : nvjpeg2kStreamParseNotFound;
    return func_ptr(handle, data, length, save_metadata, save_stream, stream_handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kStreamGetImageInfoNotFound(nvjpeg2kStream_t, nvjpeg2kImageInfo_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kStreamGetImageInfo(nvjpeg2kStream_t stream_handle, nvjpeg2kImageInfo_t* image_info) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kStream_t, nvjpeg2kImageInfo_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamGetImageInfo")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamGetImageInfo")) : nvjpeg2kStreamGetImageInfoNotFound;
    return func_ptr(stream_handle, image_info);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kStreamGetImageComponentInfoNotFound(nvjpeg2kStream_t, nvjpeg2kImageComponentInfo_t*, uint32_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kStreamGetImageComponentInfo(nvjpeg2kStream_t stream_handle, nvjpeg2kImageComponentInfo_t* component_info, uint32_t component_id) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kStream_t, nvjpeg2kImageComponentInfo_t*, uint32_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamGetImageComponentInfo")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kStreamGetImageComponentInfo")) : nvjpeg2kStreamGetImageComponentInfoNotFound;
    return func_ptr(stream_handle, component_info, component_id);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeParamsCreateNotFound(nvjpeg2kDecodeParams_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeParamsCreate(nvjpeg2kDecodeParams_t* decode_params) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeParams_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsCreate")) : nvjpeg2kDecodeParamsCreateNotFound;
    return func_ptr(decode_params);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeParamsDestroyNotFound(nvjpeg2kDecodeParams_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeParamsDestroy(nvjpeg2kDecodeParams_t decode_params) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeParams_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsDestroy")) : nvjpeg2kDecodeParamsDestroyNotFound;
    return func_ptr(decode_params);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeParamsSetDecodeAreaNotFound(nvjpeg2kDecodeParams_t, uint32_t, uint32_t, uint32_t, uint32_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeParamsSetDecodeArea(nvjpeg2kDecodeParams_t decode_params, uint32_t start_x, uint32_t end_x, uint32_t start_y, uint32_t end_y) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeParams_t, uint32_t, uint32_t, uint32_t, uint32_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetDecodeArea")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetDecodeArea")) : nvjpeg2kDecodeParamsSetDecodeAreaNotFound;
    return func_ptr(decode_params, start_x, end_x, start_y, end_y);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeParamsSetRGBOutputNotFound(nvjpeg2kDecodeParams_t, int32_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeParamsSetRGBOutput(nvjpeg2kDecodeParams_t decode_params, int32_t force_rgb) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeParams_t, int32_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetRGBOutput")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetRGBOutput")) : nvjpeg2kDecodeParamsSetRGBOutputNotFound;
    return func_ptr(decode_params, force_rgb);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeParamsSetOutputFormatNotFound(nvjpeg2kDecodeParams_t, nvjpeg2kImageFormat_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeParamsSetOutputFormat(nvjpeg2kDecodeParams_t decode_params, nvjpeg2kImageFormat_t format) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kDecodeParams_t, nvjpeg2kImageFormat_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetOutputFormat")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeParamsSetOutputFormat")) : nvjpeg2kDecodeParamsSetOutputFormatNotFound;
    return func_ptr(decode_params, format);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeImageNotFound(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t, nvjpeg2kStream_t, nvjpeg2kDecodeParams_t, nvjpeg2kImage_t*, cudaStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeImage(nvjpeg2kHandle_t handle, nvjpeg2kDecodeState_t decode_state, nvjpeg2kStream_t jpeg2k_stream, nvjpeg2kDecodeParams_t decode_params, nvjpeg2kImage_t* decode_output, cudaStream_t stream) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t, nvjpeg2kStream_t, nvjpeg2kDecodeParams_t, nvjpeg2kImage_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeImage")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeImage")) : nvjpeg2kDecodeImageNotFound;
    return func_ptr(handle, decode_state, jpeg2k_stream, decode_params, decode_output, stream);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kDecodeTileNotFound(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t, nvjpeg2kStream_t, nvjpeg2kDecodeParams_t, uint32_t, uint32_t, nvjpeg2kImage_t*, cudaStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kDecodeTile(nvjpeg2kHandle_t handle, nvjpeg2kDecodeState_t decode_state, nvjpeg2kStream_t jpeg2k_stream, nvjpeg2kDecodeParams_t decode_params, uint32_t tile_id, uint32_t num_res_levels, nvjpeg2kImage_t* decode_output, cudaStream_t stream) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kHandle_t, nvjpeg2kDecodeState_t, nvjpeg2kStream_t, nvjpeg2kDecodeParams_t, uint32_t, uint32_t, nvjpeg2kImage_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeTile")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kDecodeTile")) : nvjpeg2kDecodeTileNotFound;
    return func_ptr(handle, decode_state, jpeg2k_stream, decode_params, tile_id, num_res_levels, decode_output, stream);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncoderCreateSimpleNotFound(nvjpeg2kEncoder_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncoderCreateSimple(nvjpeg2kEncoder_t* enc_handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncoder_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncoderCreateSimple")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncoderCreateSimple")) : nvjpeg2kEncoderCreateSimpleNotFound;
    return func_ptr(enc_handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncoderDestroyNotFound(nvjpeg2kEncoder_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncoderDestroy(nvjpeg2kEncoder_t enc_handle) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncoder_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncoderDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncoderDestroy")) : nvjpeg2kEncoderDestroyNotFound;
    return func_ptr(enc_handle);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeStateCreateNotFound(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeStateCreate(nvjpeg2kEncoder_t enc_handle, nvjpeg2kEncodeState_t* encode_state) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeStateCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeStateCreate")) : nvjpeg2kEncodeStateCreateNotFound;
    return func_ptr(enc_handle, encode_state);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeStateDestroyNotFound(nvjpeg2kEncodeState_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeStateDestroy(nvjpeg2kEncodeState_t encode_state) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeState_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeStateDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeStateDestroy")) : nvjpeg2kEncodeStateDestroyNotFound;
    return func_ptr(encode_state);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsCreateNotFound(nvjpeg2kEncodeParams_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsCreate(nvjpeg2kEncodeParams_t* encode_params) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsCreate")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsCreate")) : nvjpeg2kEncodeParamsCreateNotFound;
    return func_ptr(encode_params);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsDestroyNotFound(nvjpeg2kEncodeParams_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsDestroy(nvjpeg2kEncodeParams_t encode_params) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsDestroy")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsDestroy")) : nvjpeg2kEncodeParamsDestroyNotFound;
    return func_ptr(encode_params);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSetEncodeConfigNotFound(nvjpeg2kEncodeParams_t, nvjpeg2kEncodeConfig_t*) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsSetEncodeConfig(nvjpeg2kEncodeParams_t encode_params, nvjpeg2kEncodeConfig_t* encoder_config) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t, nvjpeg2kEncodeConfig_t*);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetEncodeConfig")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetEncodeConfig")) : nvjpeg2kEncodeParamsSetEncodeConfigNotFound;
    return func_ptr(encode_params, encoder_config);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSetQualityNotFound(nvjpeg2kEncodeParams_t, double) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsSetQuality(nvjpeg2kEncodeParams_t encode_params, double target_psnr) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t, double);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetQuality")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetQuality")) : nvjpeg2kEncodeParamsSetQualityNotFound;
    return func_ptr(encode_params, target_psnr);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSpecifyQualityNotFound(nvjpeg2kEncodeParams_t, enum nvjpeg2kQualityType, double) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2kEncodeParams_t encode_params, enum nvjpeg2kQualityType quality_type, double quality_value) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t, enum nvjpeg2kQualityType, double);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSpecifyQuality")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSpecifyQuality")) : nvjpeg2kEncodeParamsSpecifyQualityNotFound;
    return func_ptr(encode_params, quality_type, quality_value);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeParamsSetInputFormatNotFound(nvjpeg2kEncodeParams_t, nvjpeg2kImageFormat_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeParamsSetInputFormat(nvjpeg2kEncodeParams_t encode_params, nvjpeg2kImageFormat_t format) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncodeParams_t, nvjpeg2kImageFormat_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetInputFormat")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeParamsSetInputFormat")) : nvjpeg2kEncodeParamsSetInputFormatNotFound;
    return func_ptr(encode_params, format);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeNotFound(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t, const nvjpeg2kEncodeParams_t, const nvjpeg2kImage_t*, cudaStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncode(nvjpeg2kEncoder_t enc_handle, nvjpeg2kEncodeState_t encode_state, const nvjpeg2kEncodeParams_t encode_params, const nvjpeg2kImage_t* input_image, cudaStream_t stream) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t, const nvjpeg2kEncodeParams_t, const nvjpeg2kImage_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncode")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncode")) : nvjpeg2kEncodeNotFound;
    return func_ptr(enc_handle, encode_state, encode_params, input_image, stream);
}

nvjpeg2kStatus_t NVJPEG2KAPI nvjpeg2kEncodeRetrieveBitstreamNotFound(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t, unsigned char*, size_t*, cudaStream_t) {
    return NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvjpeg2kStatus_t nvjpeg2kEncodeRetrieveBitstream(nvjpeg2kEncoder_t enc_handle, nvjpeg2kEncodeState_t encode_state, unsigned char* compressed_data, size_t* length, cudaStream_t stream) {
    using FuncPtr = nvjpeg2kStatus_t (*)(nvjpeg2kEncoder_t, nvjpeg2kEncodeState_t, unsigned char*, size_t*, cudaStream_t);

    static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeRetrieveBitstream")) ? reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("nvjpeg2kEncodeRetrieveBitstream")) : nvjpeg2kEncodeRetrieveBitstreamNotFound;
    return func_ptr(enc_handle, encode_state, compressed_data, length, stream);
}
