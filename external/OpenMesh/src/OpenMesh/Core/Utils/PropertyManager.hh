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

#ifndef PROPERTYMANAGER_HH_
#define PROPERTYMANAGER_HH_

#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Utils/HandleToPropHandle.hh>
#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <sstream>
#include <stdexcept>
#include <string>

namespace OpenMesh {

/**
 * This class is intended to manage the lifecycle of properties.
 * It also defines convenience operators to access the encapsulated
 * property's value.
 *
 * Note that the second template parameter is depcretated.
 *
 * \code
 * {
 *     TriMesh mesh;
 *     auto visited = makeTemporaryProperty<VertexHandle, bool>(mesh);
 *
 *     for (auto vh : mesh.vertices()) {
 *         if (!visited[vh]) {
 *             visitComponent(mesh, vh, visited);
 *         }
 *     }
 *     // The property is automatically removed at the end of the scope
 * }
 * \endcode
 */
template<typename PROPTYPE, typename MeshT = int>
class PropertyManager {

    public:
        using Value           = typename PROPTYPE::Value;
        using value_type      = typename PROPTYPE::value_type;
        using Handle          = typename PROPTYPE::Handle;
        using Self            = PropertyManager<PROPTYPE, MeshT>;
        using Reference       = typename PROPTYPE::reference;
        using ConstReference  = typename PROPTYPE::const_reference;

    private:
        // Mesh properties (MPropHandleT<...>) are stored differently than the other properties.
        // This class implements different behavior when initializing a property or when
        // copying or swapping data from one property manager to a another one.
        template <typename PropertyManager2, typename PropHandleT>
        struct StorageT;

        // specialization for Mesh Properties
        template <typename PropertyManager2>
        struct StorageT<PropertyManager2, MPropHandleT<Value>> {
          static void initialize(PropertyManager<PROPTYPE, MeshT>& pm, const Value& initial_value ) {
            pm() = initial_value;
          }
          static void copy(const PropertyManager<PROPTYPE, MeshT>& from, PropertyManager2& to) {
            *to = *from;
          }
          static void swap(PropertyManager<PROPTYPE, MeshT>& from, PropertyManager2& to) {
            std::swap(*to, *from);
          }
          static ConstReference access_property_const(PolyConnectivity& mesh, const PROPTYPE& prop_handle, const Handle&) {
            return mesh.property(prop_handle);
          }
          static Reference access_property(PolyConnectivity& mesh, const PROPTYPE& prop_handle, const Handle&) {
            return mesh.property(prop_handle);
          }
        };

        // definition for other Mesh Properties
        template <typename PropertyManager2, typename PropHandleT>
        struct StorageT {
          static void initialize(PropertyManager<PROPTYPE, MeshT>& pm, const Value& initial_value ) {
            pm.set_range(pm.mesh_.template all_elements<Handle>(), initial_value);
          }
          static void copy(const PropertyManager& from, PropertyManager2& to) {
            from.copy_to(from.mesh_.template all_elements<Handle>(), to, to.mesh_.template all_elements<Handle>());
          }
          static void swap(PropertyManager& lhs, PropertyManager2& rhs) {
            std::swap(lhs.mesh().property(lhs.prop_).data_vector(), rhs.mesh().property(rhs.prop_).data_vector());
            // resize the property to the correct size
            lhs.mesh().property(lhs.prop_).resize(lhs.mesh().template n_elements<Handle>());
            rhs.mesh().property(rhs.prop_).resize(rhs.mesh().template n_elements<Handle>());
          }
          static ConstReference access_property_const(PolyConnectivity& mesh, const PROPTYPE& prop_handle, const Handle& handle) {
            return mesh.property(prop_handle, handle);
          }
          static Reference access_property(PolyConnectivity& mesh, const PROPTYPE& prop_handle, const Handle& handle) {
            return mesh.property(prop_handle, handle);
          }
        };

        using Storage = StorageT<Self, PROPTYPE>;

    public:

        /**
         * @deprecated Use a constructor without \p existing and check existance with hasProperty() instead.
         *
         * Constructor.
         *
         * Throws an \p std::runtime_error if \p existing is true and
         * no property named \p propname of the appropriate property type
         * exists.
         *
         * @param mesh The mesh on which to create the property.
         * @param propname The name of the property.
         * @param existing If false, a new property is created and its lifecycle is managed (i.e.
         * the property is deleted upon destruction of the PropertyManager instance). If true,
         * the instance merely acts as a convenience wrapper around an existing property with no
         * lifecycle management whatsoever.
         *
         * @see PropertyManager::getOrMakeProperty, PropertyManager::getProperty,
         * PropertyManager::makeTemporaryProperty
         */
        OM_DEPRECATED("Use the constructor without parameter 'existing' instead. Check for existance with hasProperty") // As long as this overload exists, initial value must be first parameter due to ambiguity for properties of type bool
        PropertyManager(PolyConnectivity& mesh, const char *propname, bool existing) : mesh_(mesh), retain_(existing), name_(propname) {
            if (existing) {
                if (!PropertyManager::mesh().get_property_handle(prop_, propname)) {
                    std::ostringstream oss;
                    oss << "Requested property handle \"" << propname << "\" does not exist.";
                    throw std::runtime_error(oss.str());
                }
            } else {
                PropertyManager::mesh().add_property(prop_, propname);
            }
        }

        /**
         * Constructor.
         *
         * Asks for a property with name propname and creates one if none exists. Lifetime is not managed.
         *
         * @param mesh The mesh on which to create the property.
         * @param propname The name of the property.
         */
        PropertyManager(PolyConnectivity& mesh, const char *propname) : mesh_(mesh), retain_(true), name_(propname) {
            if (!PropertyManager::mesh().get_property_handle(prop_, propname)) {
              PropertyManager::mesh().add_property(prop_, propname);
            }
        }

        /**
         * Constructor.
         *
         * Asks for a property with name propname and creates one if none exists. Lifetime is not managed.
         *
         * @param initial_value If the proeprty is newly created, it will be initialized with initial_value.
         *        If the property already existed, nothing is changes.
         * @param mesh The mesh on which to create the property.
         * @param propname The name of the property.
         */
        PropertyManager(const Value& initial_value, PolyConnectivity& mesh, const char *propname) : mesh_(mesh), retain_(true), name_(propname) {
            if (!mesh_.get_property_handle(prop_, propname)) {
              PropertyManager::mesh().add_property(prop_, propname);
              Storage::initialize(*this, initial_value);
            }
        }

        /**
         * Constructor.
         *
         * Create an anonymous property. Lifetime is managed.
         *
         * @param mesh The mesh on which to create the property.
         */
        explicit PropertyManager(const PolyConnectivity& mesh) : mesh_(mesh), retain_(false), name_("") {
            PropertyManager::mesh().add_property(prop_, name_);
        }

        /**
         * Constructor.
         *
         * Create an anonymous property. Lifetime is managed.
         *
         * @param initial_value The property will be initialized with initial_value.
         * @param mesh The mesh on which to create the property.
         */
        PropertyManager(const Value& initial_value, const PolyConnectivity& mesh) : mesh_(mesh), retain_(false), name_("") {
            PropertyManager::mesh().add_property(prop_, name_);
            Storage::initialize(*this, initial_value);
        }

        /**
         * Constructor.
         *
         * Create a wrapper around an existing property. Lifetime is not managed.
         *
         * @param mesh The mesh on which to create the property.
         * @param property_handle Handle to an existing property that should be wrapped.
         */
        PropertyManager(PolyConnectivity& mesh, PROPTYPE property_handle) : mesh_(mesh), prop_(property_handle), retain_(true), name_() {
        }

        PropertyManager() = delete;

        PropertyManager(const PropertyManager& rhs)
          :
            mesh_(rhs.mesh_),
            prop_(),
            retain_(rhs.retain_),
            name_(rhs.name_)
        {
          if (rhs.retain_) // named property -> create a property manager referring to the same
          {
            prop_ = rhs.prop_;
          }
          else // unnamed property -> create a property manager refering to a new property and copy the contents
          {
            PropertyManager::mesh().add_property(prop_, name_);
            Storage::copy(rhs, *this);
          }
        }


        /**
         * Create property manager referring to a copy of the current property.
         * This can be used to explicitely create a copy of a named property. The cloned property
         * will be unnamed.
         */
        PropertyManager clone()
        {
          PropertyManager result(this->mesh());
          Storage::copy(*this, result);
          return result;
        }

        PropertyManager& operator=(const PropertyManager& rhs)
        {
          if (&mesh_ == &rhs.mesh_ && prop_ == rhs.prop_)
            ; // nothing to do
          else
            Storage::copy(rhs, *this);
          return *this;
        }

        ~PropertyManager() {
            deleteProperty();
        }

        void swap(PropertyManager &rhs) {
          // swap the data stored in the properties
          Storage::swap(rhs, *this);
        }

        static bool propertyExists(const PolyConnectivity &mesh, const char *propname) {
            PROPTYPE dummy;
            return mesh.get_property_handle(dummy, propname);
        }

        bool isValid() const { return prop_.is_valid(); }
        operator bool() const { return isValid(); }

        const PROPTYPE &getRawProperty() const { return prop_; }

        const std::string &getName() const { return name_; }

        /**
         * Get the mesh corresponding to the property.
         *
         * If you use PropertyManager without second template parameter (recommended)
         * you need to specify the actual mesh type when using this function, e.g.:
         * \code
         * {
         *     TriMesh mesh;
         *     auto visited = VProp<bool>(mesh);
         *     TriMesh& mesh_ref = visited.getMesh<TriMesh>();
         * }
         *
         */
        template <typename MeshType >
        const MeshType& getMesh() const { return dynamic_cast<const MeshType&>(mesh_); }

        const MeshT& getMesh() const { return dynamic_cast<const MeshT&>(mesh_); }


        /**
         * @deprecated This method no longer has any effect. Instead, named properties are always retained, while unnamed ones are not
         *
         * Tells the PropertyManager whether lifetime should be managed or not.
         */
        OM_DEPRECATED("retain no longer has any effect. Instead, named properties are always retained, while unnamed ones are not.")
        void retain(bool = true) {}

        /**
         * Move constructor. Transfers ownership (delete responsibility).
         */
        PropertyManager(PropertyManager &&rhs)
          :
            mesh_(rhs.mesh_),
            prop_(rhs.prop_),
            retain_(rhs.retain_),
            name_(rhs.name_)
        {
          if (!rhs.retain_)
            rhs.prop_.invalidate(); // only invalidate unnamed properties
        }

        /**
         * Move assignment. Transfers ownership (delete responsibility).
         */
        PropertyManager& operator=(PropertyManager&& rhs)
        {
            if ((&mesh_ != &rhs.mesh_) || (prop_ != rhs.prop_))
            {
              if (rhs.retain_)
              {
                // retained properties cannot be invalidated. Copy instead
                Storage::copy(rhs, *this);
              }
              else
              {
                // swap the data stored in the properties
                Storage::swap(rhs, *this);
                // remove the property from rhs
                rhs.mesh().remove_property(rhs.prop_);
                // invalidate prop_
                rhs.prop_.invalidate();
              }
            }
            return *this;
        }

        /**
         * Create a property manager for the supplied property and mesh.
         * If the property doesn't exist, it is created. In any case,
         * lifecycle management is disabled.
         *
         * @see makePropertyManagerFromExistingOrNew
         */
        static PropertyManager createIfNotExists(PolyConnectivity &mesh, const char *propname) {
            return PropertyManager(mesh, propname);
        }

        /**
         * Like createIfNotExists() with two parameters except, if the property
         * doesn't exist, it is initialized with the supplied value over
         * the supplied range after creation. If the property already exists,
         * this method has the exact same effect as the two parameter version.
         * Lifecycle management is disabled in any case.
         *
         * @see makePropertyManagerFromExistingOrNew
         */
        template<typename PROP_VALUE, typename ITERATOR_TYPE>
        static PropertyManager createIfNotExists(PolyConnectivity &mesh, const char *propname,
                const ITERATOR_TYPE &begin, const ITERATOR_TYPE &end,
                const PROP_VALUE &init_value) {
            const bool exists = propertyExists(mesh, propname);
            PropertyManager pm(mesh, propname, exists);
            if (!exists)
                pm.set_range(begin, end, init_value);
            return std::move(pm);
        }

        /**
         * Like createIfNotExists() with two parameters except, if the property
         * doesn't exist, it is initialized with the supplied value over
         * the supplied range after creation. If the property already exists,
         * this method has the exact same effect as the two parameter version.
         * Lifecycle management is disabled in any case.
         *
         * @see makePropertyManagerFromExistingOrNew
         */
        template<typename PROP_VALUE, typename ITERATOR_RANGE>
        static PropertyManager createIfNotExists(PolyConnectivity &mesh, const char *propname,
                const ITERATOR_RANGE &range, const PROP_VALUE &init_value) {
            return createIfNotExists(
                    mesh, propname, range.begin(), range.end(), init_value);
        }


        /**
         * Access the value of the encapsulated mesh property.
         *
         * Example:
         * @code
         * PolyMesh m;
         * auto description = getOrMakeProperty<void, std::string>(m, "description");
         * *description = "This is a very nice mesh.";
         * @endcode
         *
         * @note This method is only used for mesh properties.
         */
        typename PROPTYPE::reference& operator*() {
            return mesh().mproperty(prop_)[0];
        }

        /**
         * Access the value of the encapsulated mesh property.
         *
         * Example:
         * @code
         * PolyMesh m;
         * auto description = getProperty<void, std::string>(m, "description");
         * std::cout << *description << std::endl;
         * @endcode
         *
         * @note This method is only used for mesh properties.
         */
        typename PROPTYPE::const_reference& operator*() const {
            return mesh().mproperty(prop_)[0];
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        inline typename PROPTYPE::reference operator[] (Handle handle) {
            return mesh().property(prop_, handle);
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        inline typename PROPTYPE::const_reference operator[] (const Handle& handle) const {
            return mesh().property(prop_, handle);
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        inline typename PROPTYPE::reference operator() (const Handle& handle = Handle()) {
//            return mesh().property(prop_, handle);
            return Storage::access_property(mesh(), prop_, handle);
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        inline typename PROPTYPE::const_reference operator() (const Handle& handle = Handle()) const {
//            return mesh().property(prop_, handle);
            return Storage::access_property_const(mesh(), prop_, handle);
        }

        /**
         * Conveniently set the property for an entire range of values.
         *
         * Examples:
         * \code
         * MeshT mesh;
         * PropertyManager<VPropHandleT<double>> distance(
         *     mesh, "distance.plugin-example.i8.informatik.rwth-aachen.de");
         * distance.set_range(
         *     mesh.vertices_begin(), mesh.vertices_end(),
         *     std::numeric_limits<double>::infinity());
         * \endcode
         * or
         * \code
         * MeshT::VertexHandle vh;
         * distance.set_range(
         *     mesh.vv_begin(vh), mesh.vv_end(vh),
         *     std::numeric_limits<double>::infinity());
         * \endcode
         *
         * @param begin Start iterator. Needs to dereference to HandleType.
         * @param end End iterator. (Exclusive.)
         * @param value The value the range will be set to.
         */
        template<typename HandleTypeIterator, typename PROP_VALUE>
        void set_range(HandleTypeIterator begin, HandleTypeIterator end,
                const PROP_VALUE &value) {
            for (; begin != end; ++begin)
                (*this)[*begin] = value;
        }

#if (defined(_MSC_VER) && (_MSC_VER >= 1800)) || __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
        template<typename HandleTypeIteratorRange, typename PROP_VALUE>
        void set_range(const HandleTypeIteratorRange &range,
                const PROP_VALUE &value) {
            set_range(range.begin(), range.end(), value);
        }
#endif

        /**
         * Conveniently transfer the values managed by one property manager
         * onto the values managed by a different property manager.
         *
         * @param begin Start iterator. Needs to dereference to HandleType. Will
         * be used with this property manager.
         * @param end End iterator. (Exclusive.) Will be used with this property
         * manager.
         * @param dst_propmanager The destination property manager.
         * @param dst_begin Start iterator. Needs to dereference to the
         * HandleType of dst_propmanager. Will be used with dst_propmanager.
         * @param dst_end End iterator. (Exclusive.)
         * Will be used with dst_propmanager. Used to double check the bounds.
         */
        template<typename HandleTypeIterator, typename PropertyManager2,
                 typename HandleTypeIterator2>
        void copy_to(HandleTypeIterator begin, HandleTypeIterator end,
                PropertyManager2 &dst_propmanager,
                HandleTypeIterator2 dst_begin, HandleTypeIterator2 dst_end) const {

            for (; begin != end && dst_begin != dst_end; ++begin, ++dst_begin) {
                dst_propmanager[*dst_begin] = (*this)[*begin];
            }
        }

        template<typename RangeType, typename PropertyManager2,
                 typename RangeType2>
        void copy_to(const RangeType &range,
                PropertyManager2 &dst_propmanager,
                const RangeType2 &dst_range) const {
            copy_to(range.begin(), range.end(), dst_propmanager,
                    dst_range.begin(), dst_range.end());
        }


        /**
         * Copy the values of a property from a source range to
         * a target range. The source range must not be smaller than the
         * target range.
         *
         * @param prop_name Name of the property to copy. Must exist on the
         * source mesh. Will be created on the target mesh if it doesn't exist.
         *
         * @param src_mesh Source mesh from which to copy.
         * @param src_range Source range which to copy. Must not be smaller than
         * dst_range.
         * @param dst_mesh Destination mesh on which to copy.
         * @param dst_range Destination range.
         */
        template<typename RangeType, typename RangeType2>
        static void copy(const char *prop_name,
                PolyConnectivity &src_mesh, const RangeType &src_range,
                PolyConnectivity &dst_mesh, const RangeType2 &dst_range) {

            typedef OpenMesh::PropertyManager<PROPTYPE> DstPM;
            DstPM dst(DstPM::createIfNotExists(dst_mesh, prop_name));

            typedef OpenMesh::PropertyManager<PROPTYPE> SrcPM;
            SrcPM src(src_mesh, prop_name, true);

            src.copy_to(src_range, dst, dst_range);
        }

        /**
         * Mark whether this property should be stored when mesh is written
         * to a file
         *
         * @param _persistence Property will be stored iff _persistence is true
         */
        void set_persistent(bool _persistence = true)
        {
          mesh().property(getRawProperty()).set_persistent(_persistence);
        }

    private:
        void deleteProperty() {
            if (!retain_ && prop_.is_valid())
                mesh().remove_property(prop_);
        }

        PolyConnectivity& mesh() const
        {
          return const_cast<PolyConnectivity&>(mesh_);
        }

    private:
        const PolyConnectivity& mesh_;
        PROPTYPE prop_;
        bool retain_;
        std::string name_;
};

template <typename PropertyT>
class ConstPropertyViewer
{
public:
  using Value      = typename PropertyT::Value;
  using value_type = typename PropertyT::value_type;
  using Handle     = typename PropertyT::Handle;

  ConstPropertyViewer(const PolyConnectivity& mesh, const PropertyT& property_handle)
    :
      mesh_(mesh),
      prop_(property_handle)
  {}

  inline const typename PropertyT::const_reference operator() (const Handle& handle)
  {
    return mesh_.property(prop_, handle);
  }

  inline const typename PropertyT::const_reference operator[] (const Handle& handle)
  {
    return mesh_.property(prop_, handle);
  }

private:
  const PolyConnectivity& mesh_;
  PropertyT prop_;
};

/** @relates PropertyManager
 *
 * @deprecated Temporary properties should not have a name.
 *
 * Creates a new property whose lifetime is limited to the current scope.
 *
 * Used for temporary properties. Shadows any existing properties of
 * matching name and type.
 *
 * Example:
 * @code
 * PolyMesh m;
 * {
 *     auto is_quad = makeTemporaryProperty<FaceHandle, bool>(m);
 *     for (auto& fh : m.faces()) {
 *         is_quad[fh] = (m.valence(fh) == 4);
 *     }
 *     // The property is automatically removed from the mesh at the end of the scope.
 * }
 * @endcode
 *
 * @param mesh The mesh on which the property is created
 * @param propname (optional) The name of the created property
 * @tparam ElementT Element type of the created property, e.g. VertexHandle, HalfedgeHandle, etc.
 * @tparam T Value type of the created property, e.g., \p double, \p int, etc.
 * @returns A PropertyManager handling the lifecycle of the property
 */
template<typename ElementT, typename T>
PropertyManager<typename HandleToPropHandle<ElementT, T>::type>
OM_DEPRECATED("Named temporary properties are deprecated. Either create a temporary without name or a non-temporary with name")
makeTemporaryProperty(PolyConnectivity &mesh, const char *propname) {
    return PropertyManager<typename HandleToPropHandle<ElementT, T>::type>(mesh, propname, false);
}

/** @relates PropertyManager
 *
 * Creates a new property whose lifetime is limited to the current scope.
 *
 * Used for temporary properties. Shadows any existing properties of
 * matching name and type.
 *
 * Example:
 * @code
 * PolyMesh m;
 * {
 *     auto is_quad = makeTemporaryProperty<FaceHandle, bool>(m);
 *     for (auto& fh : m.faces()) {
 *         is_quad[fh] = (m.valence(fh) == 4);
 *     }
 *     // The property is automatically removed from the mesh at the end of the scope.
 * }
 * @endcode
 *
 * @param mesh The mesh on which the property is created
 * @tparam ElementT Element type of the created property, e.g. VertexHandle, HalfedgeHandle, etc.
 * @tparam T Value type of the created property, e.g., \p double, \p int, etc.
 * @returns A PropertyManager handling the lifecycle of the property
 */
template<typename ElementT, typename T>
PropertyManager<typename HandleToPropHandle<ElementT, T>::type>
makeTemporaryProperty(PolyConnectivity &mesh) {
    return PropertyManager<typename HandleToPropHandle<ElementT, T>::type>(mesh);
}


/** @relates PropertyManager
 *
 * Tests whether a property with the given element type, value type, and name is
 * present on the given mesh.
 *
 * * Example:
 * @code
 * PolyMesh m;
 * if (hasProperty<FaceHandle, bool>(m, "is_quad")) {
 *     // We now know the property exists: getProperty won't throw.
 *     auto is_quad = getProperty<FaceHandle, bool>(m, "is_quad");
 *     // Use is_quad here.
 * }
 * @endcode
 *
 * @param mesh The mesh in question
 * @param propname The property name of the expected property
 * @tparam ElementT Element type of the expected property, e.g. VertexHandle, HalfedgeHandle, etc.
 * @tparam T Value type of the expected property, e.g., \p double, \p int, etc.
 * @tparam MeshT Type of the mesh. Can often be inferred from \p mesh
 */
template<typename ElementT, typename T>
bool
hasProperty(const PolyConnectivity &mesh, const char *propname) {
    typename HandleToPropHandle<ElementT, T>::type ph;
    return mesh.get_property_handle(ph, propname);
}

/** @relates PropertyManager
 *
 * Obtains a handle to a named property.
 *
 * Example:
 * @code
 * PolyMesh m;
 * {
 *     try {
 *         auto is_quad = getProperty<FaceHandle, bool>(m, "is_quad");
 *         // Use is_quad here.
 *     }
 *     catch (const std::runtime_error& e) {
 *         // There is no is_quad face property on the mesh.
 *     }
 * }
 * @endcode
 *
 * @pre Property with the name \p propname of matching type exists.
 * @throws std::runtime_error if no property with the name \p propname of
 * matching type exists.
 * @param mesh The mesh on which the property is created
 * @param propname The name of the created property
 * @tparam ElementT Element type of the created property, e.g. VertexHandle, HalfedgeHandle, etc.
 * @tparam T Value type of the created property, e.g., \p double, \p int, etc.
 * @returns A PropertyManager wrapping the property
 */
template<typename ElementT, typename T>
PropertyManager<typename HandleToPropHandle<ElementT, T>::type>
getProperty(PolyConnectivity &mesh, const char *propname) {
  if (!hasProperty<ElementT, T>(mesh, propname))
  {
    std::ostringstream oss;
    oss << "Requested property handle \"" << propname << "\" does not exist.";
    throw std::runtime_error(oss.str());
  }
  return PropertyManager<typename HandleToPropHandle<ElementT, T>::type>(mesh, propname);
}

/** @relates PropertyManager
 *
 * Obtains a handle to a named property if it exists or creates a new one otherwise.
 *
 * Used for creating or accessing permanent properties.
 *
 * Example:
 * @code
 * PolyMesh m;
 * {
 *     auto is_quad = getOrMakeProperty<FaceHandle, bool>(m, "is_quad");
 *     for (auto& fh : m.faces()) {
 *         is_quad[fh] = (m.valence(fh) == 4);
 *     }
 *     // The property remains on the mesh after the end of the scope.
 * }
 * {
 *     // Retrieve the property from the previous scope.
 *     auto is_quad = getOrMakeProperty<FaceHandle, bool>(m, "is_quad");
 *     // Use is_quad here.
 * }
 * @endcode
 *
 * @param mesh The mesh on which the property is created
 * @param propname The name of the created property
 * @tparam ElementT Element type of the created property, e.g. VertexHandle, HalfedgeHandle, etc.
 * @tparam T Value type of the created property, e.g., \p double, \p int, etc.
 * @returns A PropertyManager wrapping the property
 */
template<typename ElementT, typename T>
PropertyManager<typename HandleToPropHandle<ElementT, T>::type>
getOrMakeProperty(PolyConnectivity &mesh, const char *propname) {
    return PropertyManager<typename HandleToPropHandle<ElementT, T>::type>::createIfNotExists(mesh, propname);
}

/** @relates PropertyManager
 * @deprecated Use makeTemporaryProperty() instead.
 *
 * Creates a new property whose lifecycle is managed by the returned
 * PropertyManager.
 *
 * Intended for temporary properties. Shadows any existing properties of
 * matching name and type.
 */
template<typename PROPTYPE>
OM_DEPRECATED("Use makeTemporaryProperty instead.")
PropertyManager<PROPTYPE> makePropertyManagerFromNew(PolyConnectivity &mesh, const char *propname)
{
    return PropertyManager<PROPTYPE>(mesh, propname, false);
}

/** \relates PropertyManager
 * @deprecated Use getProperty() instead.
 *
 * Creates a non-owning wrapper for an existing mesh property (no lifecycle
 * management).
 *
 * Intended for convenient access.
 *
 * @pre Property with the name \p propname of matching type exists.
 * @throws std::runtime_error if no property with the name \p propname of
 * matching type exists.
 */
template<typename PROPTYPE, typename MeshT = int>
OM_DEPRECATED("Use getProperty instead.")
PropertyManager<PROPTYPE, MeshT> makePropertyManagerFromExisting(PolyConnectivity &mesh, const char *propname)
{
    return PropertyManager<PROPTYPE, MeshT>(mesh, propname, true);
}

/** @relates PropertyManager
 * @deprecated Use getOrMakeProperty() instead.
 *
 * Creates a non-owning wrapper for a mesh property (no lifecycle management).
 * If the given property does not exist, it is created.
 *
 * Intended for creating or accessing persistent properties.
 */
template<typename PROPTYPE, typename MeshT = int>
OM_DEPRECATED("Use getOrMakeProperty instead.")
PropertyManager<PROPTYPE, MeshT> makePropertyManagerFromExistingOrNew(PolyConnectivity &mesh, const char *propname)
{
    return PropertyManager<PROPTYPE, MeshT>::createIfNotExists(mesh, propname);
}

/** @relates PropertyManager
 * Like the two parameter version of makePropertyManagerFromExistingOrNew()
 * except it initializes the property with the specified value over the
 * specified range if it needs to be created. If the property already exists,
 * this function has the exact same effect as the two parameter version.
 *
 * Creates a non-owning wrapper for a mesh property (no lifecycle management).
 * If the given property does not exist, it is created.
 *
 * Intended for creating or accessing persistent properties.
 */
template<typename PROPTYPE,
    typename ITERATOR_TYPE, typename PROP_VALUE>
OM_DEPRECATED("Use getOrMakeProperty instead.")
PropertyManager<PROPTYPE> makePropertyManagerFromExistingOrNew(
        PolyConnectivity &mesh, const char *propname,
        const ITERATOR_TYPE &begin, const ITERATOR_TYPE &end,
        const PROP_VALUE &init_value) {
    return PropertyManager<PROPTYPE>::createIfNotExists(
            mesh, propname, begin, end, init_value);
}

/** @relates PropertyManager
 * Like the two parameter version of makePropertyManagerFromExistingOrNew()
 * except it initializes the property with the specified value over the
 * specified range if it needs to be created. If the property already exists,
 * this function has the exact same effect as the two parameter version.
 *
 * Creates a non-owning wrapper for a mesh property (no lifecycle management).
 * If the given property does not exist, it is created.
 *
 * Intended for creating or accessing persistent properties.
 */
template<typename PROPTYPE,
    typename ITERATOR_RANGE, typename PROP_VALUE>
OM_DEPRECATED("Use getOrMakeProperty instead.")
PropertyManager<PROPTYPE> makePropertyManagerFromExistingOrNew(
        PolyConnectivity &mesh, const char *propname,
        const ITERATOR_RANGE &range,
        const PROP_VALUE &init_value) {
    return makePropertyManagerFromExistingOrNew<PROPTYPE>(
            mesh, propname, range.begin(), range.end(), init_value);
}


/** @relates PropertyManager
 * Returns a convenience wrapper around the points property of a mesh.
 */
template<typename MeshT>
PropertyManager<OpenMesh::VPropHandleT<typename MeshT::Point>>
getPointsProperty(MeshT &mesh) {
  return PropertyManager<OpenMesh::VPropHandleT<typename MeshT::Point>>(mesh, mesh.points_property_handle());
}

/** @relates PropertyManager
 * Returns a convenience wrapper around the points property of a mesh that only allows const access.
 */
template<typename MeshT>
ConstPropertyViewer<OpenMesh::VPropHandleT<typename MeshT::Point>>
getPointsProperty(const MeshT &mesh) {
  using PropType = OpenMesh::VPropHandleT<typename MeshT::Point>;
  return ConstPropertyViewer<PropType>(mesh, mesh.points_property_handle());
}

template <typename HandleT, typename T>
using Prop = PropertyManager<typename PropHandle<HandleT>::template type<T>>;

template <typename T>
using VProp = PropertyManager<OpenMesh::VPropHandleT<T>>;

template <typename T>
using HProp = PropertyManager<OpenMesh::HPropHandleT<T>>;

template <typename T>
using EProp = PropertyManager<OpenMesh::EPropHandleT<T>>;

template <typename T>
using FProp = PropertyManager<OpenMesh::FPropHandleT<T>>;

template <typename T>
using MProp = PropertyManager<OpenMesh::MPropHandleT<T>>;


} /* namespace OpenMesh */
#endif /* PROPERTYMANAGER_HH_ */
