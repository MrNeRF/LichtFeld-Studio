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

#pragma once

#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Utils/HandleToPropHandle.hh>
#include <OpenMesh/Core/Utils/PropertyManager.hh>
#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include <OpenMesh/Core/IO/SR_store.hh>


namespace OpenMesh {

#define OM_CONCAT_IMPL(a, b) a##b
#define OM_CONCAT(a, b) OM_CONCAT_IMPL(a, b)

/** \brief Base class for property creators
 *
 * A PropertyCreator can add a named property to a mesh.
 * The type of the property is specified in the classes derived from this class.
 *
 * */
class OPENMESHDLLEXPORT PropertyCreator
{
public:

  /// The string that corresponds to the type this property creator can create
  virtual std::string type_string() = 0;

  virtual std::string type_id_string() = 0;

  /// Returns true iff the given _type_name corresponds to the string the derived class can create a property for
  bool can_you_create(const std::string &_type_name);

  /// Create a vertex property on _mesh with name _property_name
  virtual void create_vertex_property  (BaseKernel& _mesh, const std::string& _property_name) = 0;

  /// Create a halfedge property on _mesh with name _property_name
  virtual void create_halfedge_property(BaseKernel& _mesh, const std::string& _property_name) = 0;

  /// Create an edge property on _mesh with name _property_name
  virtual void create_edge_property    (BaseKernel& _mesh, const std::string& _property_name) = 0;

  /// Create a face property on _mesh with name _property_name
  virtual void create_face_property    (BaseKernel& _mesh, const std::string& _property_name) = 0;

  /// Create a mesh property on _mesh with name _property_name
  virtual void create_mesh_property    (BaseKernel& _mesh, const std::string& _property_name) = 0;


  /// Create a property for the element of type HandleT on _mesh with name _property_name
  template <typename HandleT>
  void create_property(BaseKernel& _mesh, const std::string& _property_name);

  virtual ~PropertyCreator() {}

protected:
  PropertyCreator() {}

};

template <> inline void PropertyCreator::create_property<VertexHandle>  (BaseKernel& _mesh, const std::string& _property_name) { create_vertex_property  (_mesh, _property_name); }
template <> inline void PropertyCreator::create_property<HalfedgeHandle>(BaseKernel& _mesh, const std::string& _property_name) { create_halfedge_property(_mesh, _property_name); }
template <> inline void PropertyCreator::create_property<EdgeHandle>    (BaseKernel& _mesh, const std::string& _property_name) { create_edge_property    (_mesh, _property_name); }
template <> inline void PropertyCreator::create_property<FaceHandle>    (BaseKernel& _mesh, const std::string& _property_name) { create_face_property    (_mesh, _property_name); }
template <> inline void PropertyCreator::create_property<MeshHandle>    (BaseKernel& _mesh, const std::string& _property_name) { create_mesh_property    (_mesh, _property_name); }

/// Helper class that contains the implementation of the create_<HandleT>_property methods.
/// Implementation is injected into PropertyCreatorT.
template <typename PropertyCreatorT>
class PropertyCreatorImpl : public PropertyCreator
{
public:
  std::string type_id_string() override { return get_type_name<typename PropertyCreatorT::type>(); }

  template <typename HandleT, typename PropT>
  void create_prop(BaseKernel& _mesh, const std::string& _property_name)
  {
    using PHandle = typename PropHandle<HandleT>::template type<PropT>;
    PHandle prop;
    if (!_mesh.get_property_handle(prop, _property_name))
      _mesh.add_property(prop, _property_name);
  }

  void create_vertex_property  (BaseKernel& _mesh, const std::string& _property_name) override {  create_prop<VertexHandle  , typename PropertyCreatorT::type>(_mesh, _property_name); }
  void create_halfedge_property(BaseKernel& _mesh, const std::string& _property_name) override {  create_prop<HalfedgeHandle, typename PropertyCreatorT::type>(_mesh, _property_name);}
  void create_edge_property    (BaseKernel& _mesh, const std::string& _property_name) override {  create_prop<EdgeHandle    , typename PropertyCreatorT::type>(_mesh, _property_name);}
  void create_face_property    (BaseKernel& _mesh, const std::string& _property_name) override {  create_prop<FaceHandle    , typename PropertyCreatorT::type>(_mesh, _property_name);}
  void create_mesh_property    (BaseKernel& _mesh, const std::string& _property_name) override {  create_prop<MeshHandle    , typename PropertyCreatorT::type>(_mesh, _property_name);}

  ~PropertyCreatorImpl() override {}
protected:
  PropertyCreatorImpl() {}
};

/// Actual PropertyCreators specialize this class in order to add properties of type T
namespace  {
template <typename T>
class PropertyCreatorT : public PropertyCreatorImpl<PropertyCreatorT<T>>
{
};
}

///used to register custom type, where typename.hh does provide a string for type recognition

#define OM_REGISTER_PROPERTY_TYPE(ClassName)                              \
namespace OpenMesh {                                                                           \
namespace  {  /* ensure internal linkage of class */                                           \
template <>                                                                                    \
class PropertyCreatorT<ClassName> : public PropertyCreatorImpl<PropertyCreatorT<ClassName>>    \
{                                                                                              \
public:                                                                                        \
  using type = ClassName;                                                                      \
  std::string type_string() override { return OpenMesh::IO::binary<type>::type_identifier(); } \
                                                                                               \
  PropertyCreatorT()                                                                           \
  {                                                                                            \
    PropertyCreationManager::instance().register_property_creator(this);                       \
  }                                                                                            \
  ~PropertyCreatorT() override {}                                                              \
};                                                                                             \
}                                                                                              \
/* static to ensure internal linkage of object */                                              \
static PropertyCreatorT<ClassName> OM_CONCAT(property_creator_registration_object_, __LINE__); \
}

/** \brief Class for adding properties based on strings
 *
 * The PropertyCreationManager holds all PropertyCreators and dispatches the property creation
 * to them if they are able to create a property for a given string.
 *
 * If you want to be able to store your custom properties into a file and automatically load
 * them without manually adding the property yourself you can register your type by calling the
 * OM_REGISTER_PROPERTY_TYPE(ClassName, TypeString)
 * */
class OPENMESHDLLEXPORT PropertyCreationManager
{
public:

  static PropertyCreationManager& instance();

  template <typename HandleT>
  void create_property(BaseKernel& _mesh, const std::string& _type_name, const std::string& _property_name)
  {

    auto can_create = [_type_name](OpenMesh::PropertyCreator* pc){
      return pc->can_you_create(_type_name);
    };

    std::vector<OpenMesh::PropertyCreator*>::iterator pc_iter = std::find_if(property_creators_.begin(),
                                                                        property_creators_.end(), can_create);
    if (pc_iter != property_creators_.end())
    {
      const auto& pc = *pc_iter;
      pc->create_property<HandleT>(_mesh, _property_name);
      return;
    }

    omerr() << "No property creator registered that can create a property of type " << _type_name << std::endl;
    omerr() << "You need to register your custom type using OM_REGISTER_PROPERTY_TYPE(ClassName) and declare the struct binary<ClassName>.\
                See documentation for more details." << std::endl;
    omerr() << "Adding property failed." << std::endl;
  }

  void register_property_creator(PropertyCreator* _property_creator)
  {
    for (auto pc : property_creators_)
      if (pc->type_string() == _property_creator->type_string())
      {
        if (pc->type_id_string() != _property_creator->type_id_string())
        {
          omerr() << "And it looks like you are trying to add a different type with an already existing string identification." << std::endl;
          omerr() << "Type id of existing type is " << pc->type_id_string() << " trying to add for " << _property_creator->type_id_string() << std::endl;
        }
        return;
      }
    property_creators_.push_back(_property_creator);
  }

private:

  PropertyCreationManager() {}
  ~PropertyCreationManager() {}

  std::vector<PropertyCreator*> property_creators_;
};

/** \brief Create a property with type corresponding to _type_name on _mesh with name _property_name
 *
 *  For user defined types you need to register the type using OM_REGISTER_PROPERTY_TYPE(ClassName, TypeString)
 * */
template <typename HandleT>
void create_property_from_string(BaseKernel& _mesh, const std::string& _type_name, const std::string& _property_name)
{
  PropertyCreationManager::instance().create_property<HandleT>(_mesh, _type_name, _property_name);
}

} /* namespace OpenMesh */
