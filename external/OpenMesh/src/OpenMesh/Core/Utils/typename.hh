#pragma once

/// Get an internal name for a type
/// Important, this is depends on compilers and versions, do NOT use in file formats!
/// This provides property type safety when only limited RTTI is available
/// Solution adapted from OpenVolumeMesh

#include <string>
#include <typeinfo>
#include <vector>
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Core/Geometry/VectorT.hh>

namespace OpenMesh {

template <typename T>
std::string get_type_name()
{
#ifdef _MSC_VER
    // MSVC'S type_name returns only a friendly name with name() method,
    // to get a unique name use raw_name() method instead
    return typeid(T).raw_name();
#else
    // GCC and clang curently return mangled name as name(), there is no raw_name() method
    return typeid(T).name();
#endif
}

template <typename T>
bool is_correct_type_name(const std::string& name)
{
#ifdef _MSC_VER
    // MSVC'S type_name returns only a friendly name with name() method,
    // to get a unique name use raw_name() method instead
    const char* correct_name = typeid(T).raw_name();
#else
    // GCC and clang curently return mangled name as name(), there is no raw_name() method
    const char* correct_name = typeid(T).name();
#endif
    size_t pos = 0;
    while (pos < name.size() && correct_name[pos] != 0 ) {
        if (correct_name[pos] != name[pos]) {
            return false;
        }
        ++pos;
    }
    return correct_name[pos] == 0 && pos == name.size();
}

}//namespace OpenMesh
