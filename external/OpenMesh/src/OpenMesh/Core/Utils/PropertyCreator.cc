/* ========================================================================= *
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2025, RWTH-Aachen University                 *
 *           Department of Computer Graphics and Multimedia                  *
 *                          All rights reserved.                             *
 *                            www.openmesh.org                               *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * This file is part of OpenMesh.                                            *
 *---------------------------------------------------------------------------*
 *                                                                           *
 * Redistribution and use in source and binary forms, with or without        *
 * modification, are permitted provided that the following conditions        *
 * are met:                                                                  *
 *                                                                           *
 * 1. Redistributions of source code must retain the above copyright notice, *
 *    this list of conditions and the following disclaimer.                  *
 *                                                                           *
 * 2. Redistributions in binary form must reproduce the above copyright      *
 *    notice, this list of conditions and the following disclaimer in the    *
 *    documentation and/or other materials provided with the distribution.   *
 *                                                                           *
 * 3. Neither the name of the copyright holder nor the names of its          *
 *    contributors may be used to endorse or promote products derived from   *
 *    this software without specific prior written permission.               *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *
 *                                                                           *
 * ========================================================================= */

#include "PropertyCreator.hh"

#include <OpenMesh/Core/IO/importer/BaseImporter.hh>
#include <OpenMesh/Core/Mesh/Handles.hh>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <OpenMesh/Core/IO/SR_types.hh>


namespace OpenMesh {

PropertyCreationManager& PropertyCreationManager::instance()
{
  static PropertyCreationManager manager;
  return manager;
}

bool PropertyCreator::can_you_create(const std::string& _type_name)
{
  return _type_name == type_string();
}


} /* namespace OpenMesh */

OM_REGISTER_PROPERTY_TYPE(FaceHandle)
OM_REGISTER_PROPERTY_TYPE(EdgeHandle)
OM_REGISTER_PROPERTY_TYPE(HalfedgeHandle)
OM_REGISTER_PROPERTY_TYPE(VertexHandle)
OM_REGISTER_PROPERTY_TYPE(MeshHandle)

OM_REGISTER_PROPERTY_TYPE(bool)
OM_REGISTER_PROPERTY_TYPE(float)
OM_REGISTER_PROPERTY_TYPE(double)
OM_REGISTER_PROPERTY_TYPE(long double)
OM_REGISTER_PROPERTY_TYPE(char)
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::int8_t  )
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::int16_t )
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::int32_t )
//OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::int64_t )
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::uint8_t )
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::uint16_t)
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::uint32_t)
OM_REGISTER_PROPERTY_TYPE(OpenMesh::IO::uint64_t)
OM_REGISTER_PROPERTY_TYPE(std::string)

OM_REGISTER_PROPERTY_TYPE(std::vector<bool>)
OM_REGISTER_PROPERTY_TYPE(std::vector<float>)
OM_REGISTER_PROPERTY_TYPE(std::vector<double>)
OM_REGISTER_PROPERTY_TYPE(std::vector<long double>)
OM_REGISTER_PROPERTY_TYPE(std::vector<char>)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::int8_t  >)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::int16_t >)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::int32_t >)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::uint8_t >)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::uint16_t>)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::uint32_t>)
OM_REGISTER_PROPERTY_TYPE(std::vector<OpenMesh::IO::uint64_t>)
OM_REGISTER_PROPERTY_TYPE(std::vector<std::string>)

OM_REGISTER_PROPERTY_TYPE(Vec1c);
OM_REGISTER_PROPERTY_TYPE(Vec1uc);
OM_REGISTER_PROPERTY_TYPE(Vec1s);
OM_REGISTER_PROPERTY_TYPE(Vec1us);
OM_REGISTER_PROPERTY_TYPE(Vec1i);
OM_REGISTER_PROPERTY_TYPE(Vec1ui);
OM_REGISTER_PROPERTY_TYPE(Vec1f);
OM_REGISTER_PROPERTY_TYPE(Vec1d);

OM_REGISTER_PROPERTY_TYPE(Vec2c);
OM_REGISTER_PROPERTY_TYPE(Vec2uc);
OM_REGISTER_PROPERTY_TYPE(Vec2s);
OM_REGISTER_PROPERTY_TYPE(Vec2us);
OM_REGISTER_PROPERTY_TYPE(Vec2i);
OM_REGISTER_PROPERTY_TYPE(Vec2ui);
OM_REGISTER_PROPERTY_TYPE(Vec2f);
OM_REGISTER_PROPERTY_TYPE(Vec2d);

OM_REGISTER_PROPERTY_TYPE(Vec3c);
OM_REGISTER_PROPERTY_TYPE(Vec3uc);
OM_REGISTER_PROPERTY_TYPE(Vec3s);
OM_REGISTER_PROPERTY_TYPE(Vec3us);
OM_REGISTER_PROPERTY_TYPE(Vec3i);
OM_REGISTER_PROPERTY_TYPE(Vec3ui);
OM_REGISTER_PROPERTY_TYPE(Vec3f);
OM_REGISTER_PROPERTY_TYPE(Vec3d);

OM_REGISTER_PROPERTY_TYPE(Vec4c);
OM_REGISTER_PROPERTY_TYPE(Vec4uc);
OM_REGISTER_PROPERTY_TYPE(Vec4s);
OM_REGISTER_PROPERTY_TYPE(Vec4us);
OM_REGISTER_PROPERTY_TYPE(Vec4i);
OM_REGISTER_PROPERTY_TYPE(Vec4ui);
OM_REGISTER_PROPERTY_TYPE(Vec4f);
OM_REGISTER_PROPERTY_TYPE(Vec4d);

OM_REGISTER_PROPERTY_TYPE(Vec5c);
OM_REGISTER_PROPERTY_TYPE(Vec5uc);
OM_REGISTER_PROPERTY_TYPE(Vec5s);
OM_REGISTER_PROPERTY_TYPE(Vec5us);
OM_REGISTER_PROPERTY_TYPE(Vec5i);
OM_REGISTER_PROPERTY_TYPE(Vec5ui);
OM_REGISTER_PROPERTY_TYPE(Vec5f);
OM_REGISTER_PROPERTY_TYPE(Vec5d);

OM_REGISTER_PROPERTY_TYPE(Vec6c);
OM_REGISTER_PROPERTY_TYPE(Vec6uc);
OM_REGISTER_PROPERTY_TYPE(Vec6s);
OM_REGISTER_PROPERTY_TYPE(Vec6us);
OM_REGISTER_PROPERTY_TYPE(Vec6i);
OM_REGISTER_PROPERTY_TYPE(Vec6ui);
OM_REGISTER_PROPERTY_TYPE(Vec6f);
OM_REGISTER_PROPERTY_TYPE(Vec6d);
