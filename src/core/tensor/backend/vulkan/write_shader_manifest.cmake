file(READ "${SOURCE}" _source)
string(REGEX MATCH "LFS_LOCAL_SIZE_X[ \t]*=[ \t]*([0-9]+)" _local_match "${_source}")
set(_local_size_x "${CMAKE_MATCH_1}")
string(REGEX MATCH "LFS_PUSH_CONSTANT_SIZE[ \t]*=[ \t]*([0-9]+)" _push_match "${_source}")
set(_push_constant_size "${CMAKE_MATCH_1}")
if(NOT _local_size_x OR NOT _push_constant_size)
    message(FATAL_ERROR "Shader ${SOURCE} does not declare its manifest constants")
endif()
file(WRITE "${OUTPUT}"
    "{\n"
    "  \"entry_point\": \"main\",\n"
    "  \"local_size\": [${_local_size_x}, 1, 1],\n"
    "  \"push_constant_size\": ${_push_constant_size},\n"
    "  \"required_capabilities\": [\"Int64\", \"PhysicalStorageBufferAddresses\"]\n"
    "}\n")
