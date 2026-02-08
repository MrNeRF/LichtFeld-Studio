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



#ifndef OPENMESH_POLYCONNECTIVITY_HH
#define OPENMESH_POLYCONNECTIVITY_HH

#include <OpenMesh/Core/Mesh/ArrayKernel.hh>
#include <OpenMesh/Core/Mesh/SmartRange.hh>

namespace OpenMesh
{

namespace Iterators
{
  template <class Mesh, class ValueHandle, class MemberOwner, bool (MemberOwner::*PrimitiveStatusMember)() const, size_t (MemberOwner::*PrimitiveCountMember)() const>
  class GenericIteratorT;

  template<class Mesh>
  class GenericCirculatorBaseT;

  template<typename Traits>
  class GenericCirculatorT_DEPRECATED;

  template<typename Traits, bool CW>
  class GenericCirculatorT;
}

template <typename RangeTraitT>
class EntityRange;

template<
  typename CONTAINER_T,
  typename ITER_T,
  ITER_T (CONTAINER_T::*begin_fn)() const,
  ITER_T (CONTAINER_T::*end_fn)() const>
struct RangeTraitT
{
  using CONTAINER_TYPE = CONTAINER_T;
  using ITER_TYPE = ITER_T;
  static ITER_TYPE begin(const CONTAINER_TYPE& _container) { return (_container.*begin_fn)(); }
  static ITER_TYPE end(const CONTAINER_TYPE& _container)   { return (_container.*end_fn)(); }
};


template <typename CirculatorRangeTraitT>
class CirculatorRange;

template<
    typename CONTAINER_T,
    typename ITER_T,
    typename CENTER_ENTITY_T,
    typename TO_ENTITY_T,
    ITER_T (CONTAINER_T::*begin_fn)(CENTER_ENTITY_T) const,
    ITER_T (CONTAINER_T::*end_fn)(CENTER_ENTITY_T) const>
struct CirculatorRangeTraitT
{
  using CONTAINER_TYPE = CONTAINER_T;
  using ITER_TYPE = ITER_T;
  using CENTER_ENTITY_TYPE = CENTER_ENTITY_T;
  using TO_ENTITYE_TYPE = TO_ENTITY_T;
  static ITER_TYPE begin(const CONTAINER_TYPE& _container, CENTER_ENTITY_TYPE _ce)   { return (_container.*begin_fn)(_ce); }
  static ITER_TYPE begin(const CONTAINER_TYPE& _container, HalfedgeHandle _heh, int) { return ITER_TYPE(_container, _heh); }
  static ITER_TYPE end(const CONTAINER_TYPE& _container, CENTER_ENTITY_TYPE _ce)     { return (_container.*end_fn)(_ce); }
  static ITER_TYPE end(const CONTAINER_TYPE& _container, HalfedgeHandle _heh, int)   { return ITER_TYPE(_container, _heh, true); }
};

struct SmartVertexHandle;
struct SmartHalfedgeHandle;
struct SmartEdgeHandle;
struct SmartFaceHandle;

/** \brief Connectivity Class for polygonal meshes
*/
class OPENMESHDLLEXPORT PolyConnectivity : public ArrayKernel
{
public:
  /// \name Mesh Handles
  //@{
  /// Invalid handle
  static const VertexHandle                           InvalidVertexHandle;
  /// Invalid handle
  static const HalfedgeHandle                         InvalidHalfedgeHandle;
  /// Invalid handle
  static const EdgeHandle                             InvalidEdgeHandle;
  /// Invalid handle
  static const FaceHandle                             InvalidFaceHandle;
  //@}

  typedef PolyConnectivity                            This;

  //--- iterators ---

  /** \name Mesh Iterators
      Refer to OpenMesh::Mesh::Iterators or \ref mesh_iterators for
      documentation.
  */
  //@{
  /// Linear iterator
  typedef Iterators::GenericIteratorT<This, This::VertexHandle, ArrayKernel , &ArrayKernel::has_vertex_status, &ArrayKernel::n_vertices> VertexIter;
  typedef Iterators::GenericIteratorT<This, This::HalfedgeHandle, ArrayKernel , &ArrayKernel::has_halfedge_status, &ArrayKernel::n_halfedges> HalfedgeIter;
  typedef Iterators::GenericIteratorT<This, This::EdgeHandle, ArrayKernel , &ArrayKernel::has_edge_status, &ArrayKernel::n_edges> EdgeIter;
  typedef Iterators::GenericIteratorT<This, This::FaceHandle, ArrayKernel , &ArrayKernel::has_face_status, &ArrayKernel::n_faces> FaceIter;

  typedef VertexIter ConstVertexIter;
  typedef HalfedgeIter ConstHalfedgeIter;
  typedef EdgeIter ConstEdgeIter;
  typedef FaceIter ConstFaceIter;
  //@}

  //--- circulators ---

  /** \name Mesh Circulators
      Refer to OpenMesh::Mesh::Iterators or \ref mesh_iterators
      for documentation.
  */
  //@{

  /*
   * Vertex-centered circulators
   */

  struct VertexVertexTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::VertexHandle;
    using ValueHandle = This::VertexHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return _mesh->to_vertex_handle(_heh);}
  };


  /**
   * Enumerates 1-ring vertices in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<VertexVertexTraits> VertexVertexIter;
  typedef Iterators::GenericCirculatorT<VertexVertexTraits, true> VertexVertexCWIter;

  /**
   * Enumerates 1-ring vertices in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<VertexVertexTraits, false>  VertexVertexCCWIter;


  struct VertexHalfedgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::VertexHandle;
    using ValueHandle = This::HalfedgeHandle;
    static ValueHandle toHandle(const Mesh* const /*_mesh*/, This::HalfedgeHandle _heh) { return _heh;}
  };

  /**
   * Enumerates outgoing half edges in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<VertexHalfedgeTraits> VertexOHalfedgeIter;
  typedef Iterators::GenericCirculatorT<VertexHalfedgeTraits, true> VertexOHalfedgeCWIter;

  /**
   * Enumerates outgoing half edges in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<VertexHalfedgeTraits, false> VertexOHalfedgeCCWIter;

  struct VertexOppositeHalfedgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::VertexHandle;
    using ValueHandle = This::HalfedgeHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return _mesh->opposite_halfedge_handle(_heh); }
  };

  /**
   * Enumerates incoming half edges in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<VertexOppositeHalfedgeTraits> VertexIHalfedgeIter;
  typedef Iterators::GenericCirculatorT<VertexOppositeHalfedgeTraits, true> VertexIHalfedgeCWIter;

  /**
   * Enumerates incoming half edges in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<VertexOppositeHalfedgeTraits, false> VertexIHalfedgeCCWIter;


  struct VertexFaceTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::VertexHandle;
    using ValueHandle = This::FaceHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->face_handle(_heh); }
  };

  /**
   * Enumerates incident faces in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<VertexFaceTraits> VertexFaceIter;
  typedef Iterators::GenericCirculatorT<VertexFaceTraits, true> VertexFaceCWIter;

  /**
   * Enumerates incident faces in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<VertexFaceTraits, false> VertexFaceCCWIter;


  struct VertexEdgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::VertexHandle;
    using ValueHandle = This::EdgeHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->edge_handle(_heh); }
  };

  /**
   * Enumerates incident edges in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<VertexEdgeTraits> VertexEdgeIter;
  typedef Iterators::GenericCirculatorT<VertexEdgeTraits, true> VertexEdgeCWIter;
  /**
   * Enumerates incident edges in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<VertexEdgeTraits, false> VertexEdgeCCWIter;


  struct FaceHalfedgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::FaceHandle;
    using ValueHandle = This::HalfedgeHandle;
    static ValueHandle toHandle(const Mesh* const /*_mesh*/, This::HalfedgeHandle _heh) { return _heh; }
  };

  /**
   * Identical to #FaceHalfedgeIter. God knows why this typedef exists.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<FaceHalfedgeTraits> HalfedgeLoopIter;
  typedef Iterators::GenericCirculatorT<FaceHalfedgeTraits, false> HalfedgeLoopCWIter;
  /**
   * Identical to #FaceHalfedgeIter. God knows why this typedef exists.
   */
  typedef Iterators::GenericCirculatorT<FaceHalfedgeTraits, true> HalfedgeLoopCCWIter;

  typedef VertexVertexIter        ConstVertexVertexIter;
  typedef VertexVertexCWIter      ConstVertexVertexCWIter;
  typedef VertexVertexCCWIter     ConstVertexVertexCCWIter;
  typedef VertexOHalfedgeIter     ConstVertexOHalfedgeIter;
  typedef VertexOHalfedgeCWIter   ConstVertexOHalfedgeCWIter;
  typedef VertexOHalfedgeCCWIter  ConstVertexOHalfedgeCCWIter;
  typedef VertexIHalfedgeIter     ConstVertexIHalfedgeIter;
  typedef VertexIHalfedgeCWIter   ConstVertexIHalfedgeCWIter;
  typedef VertexIHalfedgeCCWIter  ConstVertexIHalfedgeCCWIter;
  typedef VertexFaceIter          ConstVertexFaceIter;
  typedef VertexFaceCWIter        ConstVertexFaceCWIter;
  typedef VertexFaceCCWIter       ConstVertexFaceCCWIter;
  typedef VertexEdgeIter          ConstVertexEdgeIter;
  typedef VertexEdgeCWIter        ConstVertexEdgeCWIter;
  typedef VertexEdgeCCWIter       ConstVertexEdgeCCWIter;

  /*
   * Face-centered circulators
   */

  struct FaceVertexTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::FaceHandle;
    using ValueHandle = This::VertexHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->to_vertex_handle(_heh); }
  };

  /**
   * Enumerate incident vertices in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<FaceVertexTraits> FaceVertexIter;
  typedef Iterators::GenericCirculatorT<FaceVertexTraits, true> FaceVertexCCWIter;

  /**
   * Enumerate incident vertices in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<FaceVertexTraits, false> FaceVertexCWIter;

  /**
   * Enumerate incident half edges in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<FaceHalfedgeTraits> FaceHalfedgeIter;
  typedef Iterators::GenericCirculatorT<FaceHalfedgeTraits, true> FaceHalfedgeCCWIter;

  /**
   * Enumerate incident half edges in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<FaceHalfedgeTraits, false> FaceHalfedgeCWIter;


  struct FaceEdgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::FaceHandle;
    using ValueHandle = This::EdgeHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->edge_handle(_heh); }
  };

  /**
   * Enumerate incident edges in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<FaceEdgeTraits> FaceEdgeIter;
  typedef Iterators::GenericCirculatorT<FaceEdgeTraits, true> FaceEdgeCCWIter;

  /**
   * Enumerate incident edges in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<FaceEdgeTraits, false> FaceEdgeCWIter;


  struct FaceFaceTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::FaceHandle;
    using ValueHandle = This::FaceHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->face_handle(_mesh->opposite_halfedge_handle(_heh)); }
  };

  /**
   * Enumerate adjacent faces in a counter clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<FaceFaceTraits> FaceFaceIter;
  typedef Iterators::GenericCirculatorT<FaceFaceTraits, true> FaceFaceCCWIter;

  /**
   * Enumerate adjacent faces in a clockwise fashion.
   */
  typedef Iterators::GenericCirculatorT<FaceFaceTraits, false> FaceFaceCWIter;

  typedef FaceVertexIter        ConstFaceVertexIter;
  typedef FaceVertexCWIter      ConstFaceVertexCWIter;
  typedef FaceVertexCCWIter     ConstFaceVertexCCWIter;
  typedef FaceHalfedgeIter      ConstFaceHalfedgeIter;
  typedef FaceHalfedgeCWIter    ConstFaceHalfedgeCWIter;
  typedef FaceHalfedgeCCWIter   ConstFaceHalfedgeCCWIter;
  typedef FaceEdgeIter          ConstFaceEdgeIter;
  typedef FaceEdgeCWIter        ConstFaceEdgeCWIter;
  typedef FaceEdgeCCWIter       ConstFaceEdgeCCWIter;
  typedef FaceFaceIter          ConstFaceFaceIter;
  typedef FaceFaceCWIter        ConstFaceFaceCWIter;
  typedef FaceFaceCCWIter       ConstFaceFaceCCWIter;

  /*
   * Edge-centered circulators
   */

  struct EdgeVertexTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::EdgeHandle;
    using ValueHandle = This::VertexHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->from_vertex_handle(_heh); }
  };

  /**
   * Enumerate vertices incident to an edge.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<EdgeVertexTraits> EdgeVertexIter;

  struct EdgeHalfedgeTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::EdgeHandle;
    using ValueHandle = This::HalfedgeHandle;
    static ValueHandle toHandle(const Mesh* const /* _mesh */, This::HalfedgeHandle _heh) { return _heh; }
  };

  /**
   * Enumerate the halfedges of an edge.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<EdgeHalfedgeTraits> EdgeHalfedgeIter;

  struct EdgeFaceTraits
  {
    using Mesh = This;
    using CenterEntityHandle = This::EdgeHandle;
    using ValueHandle = This::FaceHandle;
    static ValueHandle toHandle(const Mesh* const _mesh, This::HalfedgeHandle _heh) { return static_cast<const ArrayKernel*>(_mesh)->face_handle(_heh); }
  };

  /**
   * Enumerate faces incident to an edge.
   */
  typedef Iterators::GenericCirculatorT_DEPRECATED<EdgeFaceTraits> EdgeFaceIter;

  typedef EdgeVertexIter      ConstEdgeVertexIter;
  typedef EdgeHalfedgeIter    ConstEdgeHalfedgeIter;
  typedef EdgeFaceIter        ConstEdgeFaceIter;

  /*
   * Halfedge circulator
   */
  typedef HalfedgeLoopIter      ConstHalfedgeLoopIter;
  typedef HalfedgeLoopCWIter    ConstHalfedgeLoopCWIter;
  typedef HalfedgeLoopCCWIter   ConstHalfedgeLoopCCWIter;

  //@}

  // --- shortcuts

  /** \name Typedef Shortcuts
      Provided for convenience only
  */
  //@{
  /// Alias typedef
  typedef VertexHandle    VHandle;
  typedef HalfedgeHandle  HHandle;
  typedef EdgeHandle      EHandle;
  typedef FaceHandle      FHandle;

  typedef VertexIter    VIter;
  typedef HalfedgeIter  HIter;
  typedef EdgeIter      EIter;
  typedef FaceIter      FIter;

  typedef ConstVertexIter    CVIter;
  typedef ConstHalfedgeIter  CHIter;
  typedef ConstEdgeIter      CEIter;
  typedef ConstFaceIter      CFIter;

  typedef VertexVertexIter        VVIter;
  typedef VertexVertexCWIter      VVCWIter;
  typedef VertexVertexCCWIter     VVCCWIter;
  typedef VertexOHalfedgeIter     VOHIter;
  typedef VertexOHalfedgeCWIter   VOHCWIter;
  typedef VertexOHalfedgeCCWIter  VOHCCWIter;
  typedef VertexIHalfedgeIter     VIHIter;
  typedef VertexIHalfedgeCWIter   VIHICWter;
  typedef VertexIHalfedgeCCWIter  VIHICCWter;
  typedef VertexEdgeIter          VEIter;
  typedef VertexEdgeCWIter        VECWIter;
  typedef VertexEdgeCCWIter       VECCWIter;
  typedef VertexFaceIter          VFIter;
  typedef VertexFaceCWIter        VFCWIter;
  typedef VertexFaceCCWIter       VFCCWIter;
  typedef FaceVertexIter          FVIter;
  typedef FaceVertexCWIter        FVCWIter;
  typedef FaceVertexCCWIter       FVCCWIter;
  typedef FaceHalfedgeIter        FHIter;
  typedef FaceHalfedgeCWIter      FHCWIter;
  typedef FaceHalfedgeCCWIter     FHCWWIter;
  typedef FaceEdgeIter            FEIter;
  typedef FaceEdgeCWIter          FECWIter;
  typedef FaceEdgeCCWIter         FECWWIter;
  typedef FaceFaceIter            FFIter;
  typedef EdgeVertexIter          EVIter;
  typedef EdgeHalfedgeIter        EHIter;
  typedef EdgeFaceIter            EFIter;

  typedef ConstVertexVertexIter         CVVIter;
  typedef ConstVertexVertexCWIter       CVVCWIter;
  typedef ConstVertexVertexCCWIter      CVVCCWIter;
  typedef ConstVertexOHalfedgeIter      CVOHIter;
  typedef ConstVertexOHalfedgeCWIter    CVOHCWIter;
  typedef ConstVertexOHalfedgeCCWIter   CVOHCCWIter;
  typedef ConstVertexIHalfedgeIter      CVIHIter;
  typedef ConstVertexIHalfedgeCWIter    CVIHCWIter;
  typedef ConstVertexIHalfedgeCCWIter   CVIHCCWIter;
  typedef ConstVertexEdgeIter           CVEIter;
  typedef ConstVertexEdgeCWIter         CVECWIter;
  typedef ConstVertexEdgeCCWIter        CVECCWIter;
  typedef ConstVertexFaceIter           CVFIter;
  typedef ConstVertexFaceCWIter         CVFCWIter;
  typedef ConstVertexFaceCCWIter        CVFCCWIter;
  typedef ConstFaceVertexIter           CFVIter;
  typedef ConstFaceVertexCWIter         CFVCWIter;
  typedef ConstFaceVertexCCWIter        CFVCCWIter;
  typedef ConstFaceHalfedgeIter         CFHIter;
  typedef ConstFaceHalfedgeCWIter       CFHCWIter;
  typedef ConstFaceHalfedgeCCWIter      CFHCCWIter;
  typedef ConstFaceEdgeIter             CFEIter;
  typedef ConstFaceEdgeCWIter           CFECWIter;
  typedef ConstFaceEdgeCCWIter          CFECCWIter;
  typedef ConstFaceFaceIter             CFFIter;
  typedef ConstFaceFaceCWIter           CFFCWIter;
  typedef ConstFaceFaceCCWIter          CFFCCWIter;
  typedef ConstEdgeVertexIter           CEVIter;
  typedef ConstEdgeHalfedgeIter         CEHIter;
  typedef ConstEdgeFaceIter             CEFIter;
  //@}

public:

  PolyConnectivity()  {}
  virtual ~PolyConnectivity() override {}

  inline static bool is_triangles()
  { return false; }

  /** assign_connectivity() method. See ArrayKernel::assign_connectivity()
      for more details. */
  inline void assign_connectivity(const PolyConnectivity& _other)
  { ArrayKernel::assign_connectivity(_other); }
  
  /** \name Adding items to a mesh
  */
  //@{

  /// Add a new vertex 
  inline SmartVertexHandle add_vertex();

  /** \brief Add and connect a new face
  *
  * Create a new face consisting of the vertices provided by the vertex handle vector.
  * (The vertices have to be already added to the mesh by add_vertex)
  *
  * @param _vhandles sorted list of vertex handles (also defines order in which the vertices are added to the face)
  */
  SmartFaceHandle add_face(const std::vector<VertexHandle>& _vhandles);

  /** \brief Add and connect a new face
  *
  * Create a new face consisting of the vertices provided by the vertex handle vector.
  * (The vertices have to be already added to the mesh by add_vertex)
  *
  * @param _vhandles sorted list of vertex handles (also defines order in which the vertices are added to the face)
  */
  SmartFaceHandle add_face(const std::vector<SmartVertexHandle>& _vhandles);
 
   
  /** \brief Add and connect a new face
  *
  * Create a new face consisting of three vertices provided by the handles.
  * (The vertices have to be already added to the mesh by add_vertex)
  *
  * @param _vh0 First  vertex handle
  * @param _vh1 Second vertex handle
  * @param _vh2 Third  vertex handle
  */
  SmartFaceHandle add_face(VertexHandle _vh0, VertexHandle _vh1, VertexHandle _vh2);

  /** \brief Add and connect a new face
  *
  * Create a new face consisting of four vertices provided by the handles.
  * (The vertices have to be already added to the mesh by add_vertex)
  *
  * @param _vh0 First  vertex handle
  * @param _vh1 Second vertex handle
  * @param _vh2 Third  vertex handle
  * @param _vh3 Fourth vertex handle
  */
  SmartFaceHandle add_face(VertexHandle _vh0, VertexHandle _vh1, VertexHandle _vh2, VertexHandle _vh3);
 
  /** \brief Add and connect a new face
  *
  * Create a new face consisting of vertices provided by a handle array.
  * (The vertices have to be already added to the mesh by add_vertex)
  *
  * @param _vhandles pointer to a sorted list of vertex handles (also defines order in which the vertices are added to the face)
  * @param _vhs_size number of vertex handles in the array
  */
  SmartFaceHandle add_face(const VertexHandle* _vhandles, size_t _vhs_size);

  //@}

  /// \name Deleting mesh items and other connectivity/topology modifications
  //@{

  /** Returns whether collapsing halfedge _heh is ok or would lead to
      topological inconsistencies.
      \attention This method need the Attributes::Status attribute and
      changes the \em tagged bit.  */
  bool is_collapse_ok(HalfedgeHandle _he);
    
    
  /** Mark vertex and all incident edges and faces deleted.
      Items marked deleted will be removed by garbageCollection().
      \attention Needs the Attributes::Status attribute for vertices,
      edges and faces.
  */
  void delete_vertex(VertexHandle _vh, bool _delete_isolated_vertices = true);

  /** Mark edge (two opposite halfedges) and incident faces deleted.
      Resulting isolated vertices are marked deleted if
      _delete_isolated_vertices is true. Items marked deleted will be
      removed by garbageCollection().

      \attention Needs the Attributes::Status attribute for vertices,
      edges and faces.
  */
  void delete_edge(EdgeHandle _eh, bool _delete_isolated_vertices=true);

  /** Delete face _fh and resulting degenerated empty halfedges as
      well.  Resulting isolated vertices will be deleted if
      _delete_isolated_vertices is true.

      \attention All item will only be marked to be deleted. They will
      actually be removed by calling garbage_collection().

      \attention Needs the Attributes::Status attribute for vertices,
      edges and faces.
  */
  void delete_face(FaceHandle _fh, bool _delete_isolated_vertices=true);


  //@}

  /** \name Navigation with smart handles to allow usage of old-style navigation with smart handles
  */
  //@{

  using ArrayKernel::next_halfedge_handle;
  using ArrayKernel::prev_halfedge_handle;
  using ArrayKernel::opposite_halfedge_handle;
  using ArrayKernel::ccw_rotated_halfedge_handle;
  using ArrayKernel::cw_rotated_halfedge_handle;

  inline SmartHalfedgeHandle next_halfedge_handle       (SmartHalfedgeHandle _heh) const;
  inline SmartHalfedgeHandle prev_halfedge_handle       (SmartHalfedgeHandle _heh) const;
  inline SmartHalfedgeHandle opposite_halfedge_handle   (SmartHalfedgeHandle _heh) const;
  inline SmartHalfedgeHandle ccw_rotated_halfedge_handle(SmartHalfedgeHandle _heh) const;
  inline SmartHalfedgeHandle cw_rotated_halfedge_handle (SmartHalfedgeHandle _heh) const;

  using ArrayKernel::s_halfedge_handle;
  using ArrayKernel::s_edge_handle;

  static SmartHalfedgeHandle s_halfedge_handle(SmartEdgeHandle _eh, unsigned int _i = 0);
  static SmartEdgeHandle s_edge_handle(SmartHalfedgeHandle _heh);

  using ArrayKernel::halfedge_handle;
  using ArrayKernel::edge_handle;
  using ArrayKernel::face_handle;

  inline SmartHalfedgeHandle halfedge_handle(SmartEdgeHandle _eh, unsigned int _i = 0) const;
  inline SmartHalfedgeHandle halfedge_handle(SmartFaceHandle _fh) const;
  inline SmartHalfedgeHandle halfedge_handle(SmartVertexHandle _vh) const;
  inline SmartEdgeHandle edge_handle(SmartHalfedgeHandle _heh) const;
  inline SmartFaceHandle face_handle(SmartHalfedgeHandle _heh) const;

  /// returns the face handle of the opposite halfedge
  inline SmartFaceHandle opposite_face_handle(HalfedgeHandle _heh) const;

  //@}
  
  /** \name Begin and end iterators
  */
  //@{

  /// Begin iterator for vertices
  VertexIter vertices_begin();
  /// Const begin iterator for vertices
  ConstVertexIter vertices_begin() const;
  /// End iterator for vertices
  VertexIter vertices_end();
  /// Const end iterator for vertices
  ConstVertexIter vertices_end() const;

  /// Begin iterator for halfedges
  HalfedgeIter halfedges_begin();
  /// Const begin iterator for halfedges
  ConstHalfedgeIter halfedges_begin() const;
  /// End iterator for halfedges
  HalfedgeIter halfedges_end();
  /// Const end iterator for halfedges
  ConstHalfedgeIter halfedges_end() const;

  /// Begin iterator for edges
  EdgeIter edges_begin();
  /// Const begin iterator for edges
  ConstEdgeIter edges_begin() const;
  /// End iterator for edges
  EdgeIter edges_end();
  /// Const end iterator for edges
  ConstEdgeIter edges_end() const;

  /// Begin iterator for faces
  FaceIter faces_begin();
  /// Const begin iterator for faces
  ConstFaceIter faces_begin() const;
  /// End iterator for faces
  FaceIter faces_end();
  /// Const end iterator for faces
  ConstFaceIter faces_end() const;
  //@}


  /** \name Begin for skipping iterators
  */
  //@{

  /// Begin iterator for vertices
  VertexIter vertices_sbegin();
  /// Const begin iterator for vertices
  ConstVertexIter vertices_sbegin() const;

  /// Begin iterator for halfedges
  HalfedgeIter halfedges_sbegin();
  /// Const begin iterator for halfedges
  ConstHalfedgeIter halfedges_sbegin() const;

  /// Begin iterator for edges
  EdgeIter edges_sbegin();
  /// Const begin iterator for edges
  ConstEdgeIter edges_sbegin() const;

  /// Begin iterator for faces
  FaceIter faces_sbegin();
  /// Const begin iterator for faces
  ConstFaceIter faces_sbegin() const;

  //@}

  //--- circulators ---

  /** \name Vertex, Face, and Edge circulators
  */
  //@{

  /// vertex - vertex circulator
  VertexVertexIter vv_iter(VertexHandle _vh);
  /// vertex - vertex circulator cw
  VertexVertexCWIter vv_cwiter(VertexHandle _vh);
  /// vertex - vertex circulator ccw
  VertexVertexCCWIter vv_ccwiter(VertexHandle _vh);
  /// vertex - incoming halfedge circulator
  VertexIHalfedgeIter vih_iter(VertexHandle _vh);
  /// vertex - incoming halfedge circulator cw
  VertexIHalfedgeCWIter vih_cwiter(VertexHandle _vh);
  /// vertex - incoming halfedge circulator ccw
  VertexIHalfedgeCCWIter vih_ccwiter(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator
  VertexOHalfedgeIter voh_iter(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator cw
  VertexOHalfedgeCWIter voh_cwiter(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator ccw
  VertexOHalfedgeCCWIter voh_ccwiter(VertexHandle _vh);
  /// vertex - edge circulator
  VertexEdgeIter ve_iter(VertexHandle _vh);
  /// vertex - edge circulator cw
  VertexEdgeCWIter ve_cwiter(VertexHandle _vh);
  /// vertex - edge circulator ccw
  VertexEdgeCCWIter ve_ccwiter(VertexHandle _vh);
  /// vertex - face circulator
  VertexFaceIter vf_iter(VertexHandle _vh);
  /// vertex - face circulator cw
  VertexFaceCWIter vf_cwiter(VertexHandle _vh);
  /// vertex - face circulator ccw
  VertexFaceCCWIter vf_ccwiter(VertexHandle _vh);

  /// const vertex circulator
  ConstVertexVertexIter cvv_iter(VertexHandle _vh) const;
  /// const vertex circulator cw
  ConstVertexVertexCWIter cvv_cwiter(VertexHandle _vh) const;
  /// const vertex circulator ccw
  ConstVertexVertexCCWIter cvv_ccwiter(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator
  ConstVertexIHalfedgeIter cvih_iter(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator cw
  ConstVertexIHalfedgeCWIter cvih_cwiter(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator ccw
  ConstVertexIHalfedgeCCWIter cvih_ccwiter(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator
  ConstVertexOHalfedgeIter cvoh_iter(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator cw
  ConstVertexOHalfedgeCWIter cvoh_cwiter(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator ccw
  ConstVertexOHalfedgeCCWIter cvoh_ccwiter(VertexHandle _vh) const;
  /// const vertex - edge circulator
  ConstVertexEdgeIter cve_iter(VertexHandle _vh) const;
  /// const vertex - edge circulator cw
  ConstVertexEdgeCWIter cve_cwiter(VertexHandle _vh) const;
  /// const vertex - edge circulator ccw
  ConstVertexEdgeCCWIter cve_ccwiter(VertexHandle _vh) const;
  /// const vertex - face circulator
  ConstVertexFaceIter cvf_iter(VertexHandle _vh) const;
  /// const vertex - face circulator cw
  ConstVertexFaceCWIter cvf_cwiter(VertexHandle _vh) const;
  /// const vertex - face circulator ccw
  ConstVertexFaceCCWIter cvf_ccwiter(VertexHandle _vh) const;

  /// face - vertex circulator
  FaceVertexIter fv_iter(FaceHandle _fh);
  /// face - vertex circulator cw
  FaceVertexCWIter fv_cwiter(FaceHandle _fh);
  /// face - vertex circulator ccw
  FaceVertexCCWIter fv_ccwiter(FaceHandle _fh);
  /// face - halfedge circulator
  FaceHalfedgeIter fh_iter(FaceHandle _fh);
  /// face - halfedge circulator cw
  FaceHalfedgeCWIter fh_cwiter(FaceHandle _fh);
  /// face - halfedge circulator ccw
  FaceHalfedgeCCWIter fh_ccwiter(FaceHandle _fh);
  /// face - edge circulator
  FaceEdgeIter fe_iter(FaceHandle _fh);
  /// face - edge circulator cw
  FaceEdgeCWIter fe_cwiter(FaceHandle _fh);
  /// face - edge circulator ccw
  FaceEdgeCCWIter fe_ccwiter(FaceHandle _fh);
  /// face - face circulator
  FaceFaceIter ff_iter(FaceHandle _fh);
  /// face - face circulator cw
  FaceFaceCWIter ff_cwiter(FaceHandle _fh);
  /// face - face circulator ccw
  FaceFaceCCWIter ff_ccwiter(FaceHandle _fh);

  /// const face - vertex circulator
  ConstFaceVertexIter cfv_iter(FaceHandle _fh) const;
  /// const face - vertex circulator cw
  ConstFaceVertexCWIter cfv_cwiter(FaceHandle _fh) const;
  /// const face - vertex circulator ccw
  ConstFaceVertexCCWIter cfv_ccwiter(FaceHandle _fh) const;
  /// const face - halfedge circulator
  ConstFaceHalfedgeIter cfh_iter(FaceHandle _fh) const;
  /// const face - halfedge circulator cw
  ConstFaceHalfedgeCWIter cfh_cwiter(FaceHandle _fh) const;
  /// const face - halfedge circulator ccw
  ConstFaceHalfedgeCCWIter cfh_ccwiter(FaceHandle _fh) const;
  /// const face - edge circulator
  ConstFaceEdgeIter cfe_iter(FaceHandle _fh) const;
  /// const face - edge circulator cw
  ConstFaceEdgeCWIter cfe_cwiter(FaceHandle _fh) const;
  /// const face - edge circulator ccw
  ConstFaceEdgeCCWIter cfe_ccwiter(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceIter cff_iter(FaceHandle _fh) const;
  /// const face - face circulator cw
  ConstFaceFaceCWIter cff_cwiter(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceCCWIter cff_ccwiter(FaceHandle _fh) const;

  /// edge - vertex circulator
  EdgeVertexIter ev_iter(EdgeHandle _eh);
  /// edge - halfedge circulator
  EdgeHalfedgeIter eh_iter(EdgeHandle _eh);
  /// edge - face circulator
  EdgeFaceIter ef_iter(EdgeHandle _eh);

  /// const edge - vertex circulator
  ConstEdgeVertexIter cev_iter(EdgeHandle _eh) const;
  /// const edge - halfedge circulator
  ConstEdgeHalfedgeIter ceh_iter(EdgeHandle _eh) const;
  /// const edge - face circulator
  ConstEdgeFaceIter cef_iter(EdgeHandle _eh) const;
  
  // 'begin' circulators
  
  /// vertex - vertex circulator
  VertexVertexIter vv_begin(VertexHandle _vh);
  /// vertex - vertex circulator cw
  VertexVertexCWIter vv_cwbegin(VertexHandle _vh);
  /// vertex - vertex circulator ccw
  VertexVertexCCWIter vv_ccwbegin(VertexHandle _vh);
  /// vertex - incoming halfedge circulator
  VertexIHalfedgeIter vih_begin(VertexHandle _vh);
  /// vertex - incoming halfedge circulator cw
  VertexIHalfedgeCWIter vih_cwbegin(VertexHandle _vh);
  /// vertex - incoming halfedge circulator ccw
  VertexIHalfedgeCCWIter vih_ccwbegin(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator
  VertexOHalfedgeIter voh_begin(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator cw
  VertexOHalfedgeCWIter voh_cwbegin(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator ccw
  VertexOHalfedgeCCWIter voh_ccwbegin(VertexHandle _vh);
  /// vertex - edge circulator
  VertexEdgeIter ve_begin(VertexHandle _vh);
  /// vertex - edge circulator cw
  VertexEdgeCWIter ve_cwbegin(VertexHandle _vh);
  /// vertex - edge circulator ccw
  VertexEdgeCCWIter ve_ccwbegin(VertexHandle _vh);
  /// vertex - face circulator
  VertexFaceIter vf_begin(VertexHandle _vh);
  /// vertex - face circulator cw
  VertexFaceCWIter vf_cwbegin(VertexHandle _vh);
  /// vertex - face circulator ccw
  VertexFaceCCWIter vf_ccwbegin(VertexHandle _vh);


  /// const vertex circulator
  ConstVertexVertexIter cvv_begin(VertexHandle _vh) const;
  /// const vertex circulator cw
  ConstVertexVertexCWIter cvv_cwbegin(VertexHandle _vh) const;
  /// const vertex circulator ccw
  ConstVertexVertexCCWIter cvv_ccwbegin(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator
  ConstVertexIHalfedgeIter cvih_begin(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator cw
  ConstVertexIHalfedgeCWIter cvih_cwbegin(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator ccw
  ConstVertexIHalfedgeCCWIter cvih_ccwbegin(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator
  ConstVertexOHalfedgeIter cvoh_begin(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator cw
  ConstVertexOHalfedgeCWIter cvoh_cwbegin(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator ccw
  ConstVertexOHalfedgeCCWIter cvoh_ccwbegin(VertexHandle _vh) const;
  /// const vertex - edge circulator
  ConstVertexEdgeIter cve_begin(VertexHandle _vh) const;
  /// const vertex - edge circulator cw
  ConstVertexEdgeCWIter cve_cwbegin(VertexHandle _vh) const;
  /// const vertex - edge circulator ccw
  ConstVertexEdgeCCWIter cve_ccwbegin(VertexHandle _vh) const;
  /// const vertex - face circulator
  ConstVertexFaceIter cvf_begin(VertexHandle _vh) const;
  /// const vertex - face circulator cw
  ConstVertexFaceCWIter cvf_cwbegin(VertexHandle _vh) const;
  /// const vertex - face circulator ccw
  ConstVertexFaceCCWIter cvf_ccwbegin(VertexHandle _vh) const;

  /// face - vertex circulator
  FaceVertexIter fv_begin(FaceHandle _fh);
  /// face - vertex circulator cw
  FaceVertexCWIter fv_cwbegin(FaceHandle _fh);
  /// face - vertex circulator ccw
  FaceVertexCCWIter fv_ccwbegin(FaceHandle _fh);
  /// face - halfedge circulator
  FaceHalfedgeIter fh_begin(FaceHandle _fh);
  /// face - halfedge circulator cw
  FaceHalfedgeCWIter fh_cwbegin(FaceHandle _fh);
  /// face - halfedge circulator ccw
  FaceHalfedgeCCWIter fh_ccwbegin(FaceHandle _fh);
  /// face - edge circulator
  FaceEdgeIter fe_begin(FaceHandle _fh);
  /// face - edge circulator cw
  FaceEdgeCWIter fe_cwbegin(FaceHandle _fh);
  /// face - edge circulator ccw
  FaceEdgeCCWIter fe_ccwbegin(FaceHandle _fh);
  /// face - face circulator
  FaceFaceIter ff_begin(FaceHandle _fh);
  /// face - face circulator cw
  FaceFaceCWIter ff_cwbegin(FaceHandle _fh);
  /// face - face circulator ccw
  FaceFaceCCWIter ff_ccwbegin(FaceHandle _fh);
  /// halfedge circulator
  HalfedgeLoopIter hl_begin(HalfedgeHandle _heh);
  /// halfedge circulator
  HalfedgeLoopCWIter hl_cwbegin(HalfedgeHandle _heh);
  /// halfedge circulator ccw
  HalfedgeLoopCCWIter hl_ccwbegin(HalfedgeHandle _heh);

  /// const face - vertex circulator
  ConstFaceVertexIter cfv_begin(FaceHandle _fh) const;
  /// const face - vertex circulator cw
  ConstFaceVertexCWIter cfv_cwbegin(FaceHandle _fh) const;
  /// const face - vertex circulator ccw
  ConstFaceVertexCCWIter cfv_ccwbegin(FaceHandle _fh) const;
  /// const face - halfedge circulator
  ConstFaceHalfedgeIter cfh_begin(FaceHandle _fh) const;
  /// const face - halfedge circulator cw
  ConstFaceHalfedgeCWIter cfh_cwbegin(FaceHandle _fh) const;
  /// const face - halfedge circulator ccw
  ConstFaceHalfedgeCCWIter cfh_ccwbegin(FaceHandle _fh) const;
  /// const face - edge circulator
  ConstFaceEdgeIter cfe_begin(FaceHandle _fh) const;
  /// const face - edge circulator cw
  ConstFaceEdgeCWIter cfe_cwbegin(FaceHandle _fh) const;
  /// const face - edge circulator ccw
  ConstFaceEdgeCCWIter cfe_ccwbegin(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceIter cff_begin(FaceHandle _fh) const;
  /// const face - face circulator cw
  ConstFaceFaceCWIter cff_cwbegin(FaceHandle _fh) const;
  /// const face - face circulator ccw
  ConstFaceFaceCCWIter cff_ccwbegin(FaceHandle _fh) const;
  /// const halfedge circulator
  ConstHalfedgeLoopIter chl_begin(HalfedgeHandle _heh) const;
  /// const halfedge circulator cw
  ConstHalfedgeLoopCWIter chl_cwbegin(HalfedgeHandle _heh) const;
  /// const halfedge circulator ccw
  ConstHalfedgeLoopCCWIter chl_ccwbegin(HalfedgeHandle _heh) const;

  /// edge - vertex circulator
  EdgeVertexIter ev_begin(EdgeHandle _eh);
  /// edge - halfedge circulator
  EdgeHalfedgeIter eh_begin(EdgeHandle _eh);
  /// edge - face circulator
  EdgeFaceIter ef_begin(EdgeHandle _eh);

  /// const edge - vertex circulator
  ConstEdgeVertexIter cev_begin(EdgeHandle _eh) const;
  /// const edge - halfedge circulator
  ConstEdgeHalfedgeIter ceh_begin(EdgeHandle _eh) const;
  /// const edge - face circulator
  ConstEdgeFaceIter cef_begin(EdgeHandle _eh) const;
  
  // 'end' circulators
  
  /// vertex - vertex circulator
  VertexVertexIter vv_end(VertexHandle _vh);
  /// vertex - vertex circulator cw
  VertexVertexCWIter vv_cwend(VertexHandle _vh);
  /// vertex - vertex circulator ccw
  VertexVertexCCWIter vv_ccwend(VertexHandle _vh);
  /// vertex - incoming halfedge circulator
  VertexIHalfedgeIter vih_end(VertexHandle _vh);
  /// vertex - incoming halfedge circulator cw
  VertexIHalfedgeCWIter vih_cwend(VertexHandle _vh);
  /// vertex - incoming halfedge circulator ccw
  VertexIHalfedgeCCWIter vih_ccwend(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator
  VertexOHalfedgeIter voh_end(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator cw
  VertexOHalfedgeCWIter voh_cwend(VertexHandle _vh);
  /// vertex - outgoing halfedge circulator ccw
  VertexOHalfedgeCCWIter voh_ccwend(VertexHandle _vh);
  /// vertex - edge circulator
  VertexEdgeIter ve_end(VertexHandle _vh);
  /// vertex - edge circulator cw
  VertexEdgeCWIter ve_cwend(VertexHandle _vh);
  /// vertex - edge circulator ccw
  VertexEdgeCCWIter ve_ccwend(VertexHandle _vh);
  /// vertex - face circulator
  VertexFaceIter vf_end(VertexHandle _vh);
  /// vertex - face circulator cw
  VertexFaceCWIter vf_cwend(VertexHandle _vh);
  /// vertex - face circulator ccw
  VertexFaceCCWIter vf_ccwend(VertexHandle _vh);

  /// const vertex circulator
  ConstVertexVertexIter cvv_end(VertexHandle _vh) const;
  /// const vertex circulator cw
  ConstVertexVertexCWIter cvv_cwend(VertexHandle _vh) const;
  /// const vertex circulator ccw
  ConstVertexVertexCCWIter cvv_ccwend(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator
  ConstVertexIHalfedgeIter cvih_end(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator cw
  ConstVertexIHalfedgeCWIter cvih_cwend(VertexHandle _vh) const;
  /// const vertex - incoming halfedge circulator ccw
  ConstVertexIHalfedgeCCWIter cvih_ccwend(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator
  ConstVertexOHalfedgeIter cvoh_end(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator cw
  ConstVertexOHalfedgeCWIter cvoh_cwend(VertexHandle _vh) const;
  /// const vertex - outgoing halfedge circulator ccw
  ConstVertexOHalfedgeCCWIter cvoh_ccwend(VertexHandle _vh) const;
  /// const vertex - edge circulator
  ConstVertexEdgeIter cve_end(VertexHandle _vh) const;
  /// const vertex - edge circulator cw
  ConstVertexEdgeCWIter cve_cwend(VertexHandle _vh) const;
  /// const vertex - edge circulator ccw
  ConstVertexEdgeCCWIter cve_ccwend(VertexHandle _vh) const;
  /// const vertex - face circulator
  ConstVertexFaceIter cvf_end(VertexHandle _vh) const;
  /// const vertex - face circulator cw
  ConstVertexFaceCWIter cvf_cwend(VertexHandle _vh) const;
  /// const vertex - face circulator ccw
  ConstVertexFaceCCWIter cvf_ccwend(VertexHandle _vh) const;

  /// face - vertex circulator
  FaceVertexIter fv_end(FaceHandle _fh);
  /// face - vertex circulator cw
  FaceVertexCWIter fv_cwend(FaceHandle _fh);
  /// face - vertex circulator ccw
  FaceVertexCCWIter fv_ccwend(FaceHandle _fh);
  /// face - halfedge circulator
  FaceHalfedgeIter fh_end(FaceHandle _fh);
  /// face - halfedge circulator cw
  FaceHalfedgeCWIter fh_cwend(FaceHandle _fh);
  /// face - halfedge circulator ccw
  FaceHalfedgeCCWIter fh_ccwend(FaceHandle _fh);
  /// face - edge circulator
  FaceEdgeIter fe_end(FaceHandle _fh);
  /// face - edge circulator cw
  FaceEdgeCWIter fe_cwend(FaceHandle _fh);
  /// face - edge circulator ccw
  FaceEdgeCCWIter fe_ccwend(FaceHandle _fh);
  /// face - face circulator
  FaceFaceIter ff_end(FaceHandle _fh);
  /// face - face circulator cw
  FaceFaceCWIter ff_cwend(FaceHandle _fh);
  /// face - face circulator ccw
  FaceFaceCCWIter ff_ccwend(FaceHandle _fh);
  /// face - face circulator
  HalfedgeLoopIter hl_end(HalfedgeHandle _heh);
  /// face - face circulator cw
  HalfedgeLoopCWIter hl_cwend(HalfedgeHandle _heh);
  /// face - face circulator ccw
  HalfedgeLoopCCWIter hl_ccwend(HalfedgeHandle _heh);

  /// const face - vertex circulator
  ConstFaceVertexIter cfv_end(FaceHandle _fh) const;
  /// const face - vertex circulator cw
  ConstFaceVertexCWIter cfv_cwend(FaceHandle _fh) const;
  /// const face - vertex circulator ccw
  ConstFaceVertexCCWIter cfv_ccwend(FaceHandle _fh) const;
  /// const face - halfedge circulator
  ConstFaceHalfedgeIter cfh_end(FaceHandle _fh) const;
  /// const face - halfedge circulator cw
  ConstFaceHalfedgeCWIter cfh_cwend(FaceHandle _fh) const;
  /// const face - halfedge circulator ccw
  ConstFaceHalfedgeCCWIter cfh_ccwend(FaceHandle _fh) const;
  /// const face - edge circulator
  ConstFaceEdgeIter cfe_end(FaceHandle _fh) const;
  /// const face - edge circulator cw
  ConstFaceEdgeCWIter cfe_cwend(FaceHandle _fh) const;
  /// const face - edge circulator ccw
  ConstFaceEdgeCCWIter cfe_ccwend(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceIter cff_end(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceCWIter cff_cwend(FaceHandle _fh) const;
  /// const face - face circulator
  ConstFaceFaceCCWIter cff_ccwend(FaceHandle _fh) const;
  /// const face - face circulator
  ConstHalfedgeLoopIter chl_end(HalfedgeHandle _heh) const;
  /// const face - face circulator cw
  ConstHalfedgeLoopCWIter chl_cwend(HalfedgeHandle _heh) const;
  /// const face - face circulator ccw
  ConstHalfedgeLoopCCWIter chl_ccwend(HalfedgeHandle _heh) const;

  /// edge - vertex circulator
  EdgeVertexIter ev_end(EdgeHandle _eh);
  /// edge - halfedge circulator
  EdgeHalfedgeIter eh_end(EdgeHandle _eh);
  /// edge - face circulator
  EdgeFaceIter ef_end(EdgeHandle _eh);

  /// const edge - vertex circulator
  ConstEdgeVertexIter cev_end(EdgeHandle _eh) const;
  /// const edge - halfedge circulator
  ConstEdgeHalfedgeIter ceh_end(EdgeHandle _eh) const;
  /// const edge - face circulator
  ConstEdgeFaceIter cef_end(EdgeHandle _eh) const;

  //@}

  /** @name Range based iterators and circulators */
  //@{

  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstVertexIter,
          &PolyConnectivity::vertices_begin,
          &PolyConnectivity::vertices_end>> ConstVertexRange;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstVertexIter,
          &PolyConnectivity::vertices_sbegin,
          &PolyConnectivity::vertices_end>> ConstVertexRangeSkipping;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstHalfedgeIter,
          &PolyConnectivity::halfedges_begin,
          &PolyConnectivity::halfedges_end>> ConstHalfedgeRange;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstHalfedgeIter,
          &PolyConnectivity::halfedges_sbegin,
          &PolyConnectivity::halfedges_end>> ConstHalfedgeRangeSkipping;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstEdgeIter,
          &PolyConnectivity::edges_begin,
          &PolyConnectivity::edges_end>> ConstEdgeRange;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstEdgeIter,
          &PolyConnectivity::edges_sbegin,
          &PolyConnectivity::edges_end>> ConstEdgeRangeSkipping;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstFaceIter,
          &PolyConnectivity::faces_begin,
          &PolyConnectivity::faces_end>> ConstFaceRange;
  typedef EntityRange<RangeTraitT<
          const PolyConnectivity,
          PolyConnectivity::ConstFaceIter,
          &PolyConnectivity::faces_sbegin,
          &PolyConnectivity::faces_end>> ConstFaceRangeSkipping;


  template <typename HandleType>
  struct ElementRange;

  /**
   * @return The vertices as a range object suitable
   * for C++11 range based for loops. Will skip deleted vertices.
   */
  ConstVertexRangeSkipping vertices() const;

  /**
   * @return The vertices as a range object suitable
   * for C++11 range based for loops. Will include deleted vertices.
   */
  ConstVertexRange all_vertices() const;

  /**
   * @return The halfedges as a range object suitable
   * for C++11 range based for loops. Will skip deleted halfedges.
   */
  ConstHalfedgeRangeSkipping halfedges() const;

  /**
   * @return The halfedges as a range object suitable
   * for C++11 range based for loops. Will include deleted halfedges.
   */
  ConstHalfedgeRange all_halfedges() const;

  /**
   * @return The edges as a range object suitable
   * for C++11 range based for loops. Will skip deleted edges.
   */
  ConstEdgeRangeSkipping edges() const;

  /**
   * @return The edges as a range object suitable
   * for C++11 range based for loops. Will include deleted edges.
   */
  ConstEdgeRange all_edges() const;

  /**
   * @return The faces as a range object suitable
   * for C++11 range based for loops. Will skip deleted faces.
   */
  ConstFaceRangeSkipping faces() const;

  /**
   * @return The faces as a range object suitable
   * for C++11 range based for loops. Will include deleted faces.
   */
  ConstFaceRange all_faces() const;

  /**
   * @return The elements corresponding to the template type as a range object suitable
   * for C++11 range based for loops. Will skip deleted faces.
   */
  template <typename HandleType>
  typename ElementRange<HandleType>::RangeSkipping elements() const;

  /**
   * @return The elements corresponding to the template type as a range object suitable
   * for C++11 range based for loops. Will include deleted faces.
   */
  template <typename HandleType>
  typename ElementRange<HandleType>::Range all_elements() const;


  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexVertexIter      , VertexHandle  , VertexHandle  , &PolyConnectivity::cvv_begin     , &PolyConnectivity::cvv_end     >> ConstVertexVertexRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexIHalfedgeIter   , VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvih_begin    , &PolyConnectivity::cvih_end    >> ConstVertexIHalfedgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexOHalfedgeIter   , VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvoh_begin    , &PolyConnectivity::cvoh_end    >> ConstVertexOHalfedgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexEdgeIter        , VertexHandle  , EdgeHandle    , &PolyConnectivity::cve_begin     , &PolyConnectivity::cve_end     >> ConstVertexEdgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexFaceIter        , VertexHandle  , FaceHandle    , &PolyConnectivity::cvf_begin     , &PolyConnectivity::cvf_end     >> ConstVertexFaceRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceVertexIter        , FaceHandle    , VertexHandle  , &PolyConnectivity::cfv_begin     , &PolyConnectivity::cfv_end     >> ConstFaceVertexRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceHalfedgeIter      , FaceHandle    , HalfedgeHandle, &PolyConnectivity::cfh_begin     , &PolyConnectivity::cfh_end     >> ConstFaceHalfedgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceEdgeIter          , FaceHandle    , EdgeHandle    , &PolyConnectivity::cfe_begin     , &PolyConnectivity::cfe_end     >> ConstFaceEdgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceFaceIter          , FaceHandle    , FaceHandle    , &PolyConnectivity::cff_begin     , &PolyConnectivity::cff_end     >> ConstFaceFaceRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstEdgeVertexIter        , EdgeHandle    , VertexHandle  , &PolyConnectivity::cev_begin     , &PolyConnectivity::cev_end     >> ConstEdgeVertexRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstEdgeHalfedgeIter      , EdgeHandle    , HalfedgeHandle, &PolyConnectivity::ceh_begin     , &PolyConnectivity::ceh_end     >> ConstEdgeHalfedgeRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstEdgeFaceIter          , EdgeHandle    , FaceHandle,     &PolyConnectivity::cef_begin     , &PolyConnectivity::cef_end     >> ConstEdgeFaceRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstHalfedgeLoopIter      , HalfedgeHandle, HalfedgeHandle, &PolyConnectivity::chl_begin     , &PolyConnectivity::chl_end     >> ConstHalfedgeLoopRange;

  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexVertexCWIter    , VertexHandle  , VertexHandle  , &PolyConnectivity::cvv_cwbegin   , &PolyConnectivity::cvv_cwend   >> ConstVertexVertexCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexIHalfedgeCWIter , VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvih_cwbegin  , &PolyConnectivity::cvih_cwend  >> ConstVertexIHalfedgeCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexOHalfedgeCWIter , VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvoh_cwbegin  , &PolyConnectivity::cvoh_cwend  >> ConstVertexOHalfedgeCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexEdgeCWIter      , VertexHandle  , EdgeHandle    , &PolyConnectivity::cve_cwbegin   , &PolyConnectivity::cve_cwend   >> ConstVertexEdgeCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexFaceCWIter      , VertexHandle  , FaceHandle    , &PolyConnectivity::cvf_cwbegin   , &PolyConnectivity::cvf_cwend   >> ConstVertexFaceCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceVertexCWIter      , FaceHandle    , VertexHandle  , &PolyConnectivity::cfv_cwbegin   , &PolyConnectivity::cfv_cwend   >> ConstFaceVertexCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceHalfedgeCWIter    , FaceHandle    , HalfedgeHandle, &PolyConnectivity::cfh_cwbegin   , &PolyConnectivity::cfh_cwend   >> ConstFaceHalfedgeCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceEdgeCWIter        , FaceHandle    , EdgeHandle    , &PolyConnectivity::cfe_cwbegin   , &PolyConnectivity::cfe_cwend   >> ConstFaceEdgeCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceFaceCWIter        , FaceHandle    , FaceHandle    , &PolyConnectivity::cff_cwbegin   , &PolyConnectivity::cff_cwend   >> ConstFaceFaceCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstHalfedgeLoopCWIter    , HalfedgeHandle, HalfedgeHandle, &PolyConnectivity::chl_cwbegin   , &PolyConnectivity::chl_cwend   >> ConstHalfedgeLoopCWRange;

  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexVertexCCWIter   , VertexHandle  , VertexHandle  , &PolyConnectivity::cvv_ccwbegin  , &PolyConnectivity::cvv_ccwend  >> ConstVertexVertexCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexIHalfedgeCCWIter, VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvih_ccwbegin , &PolyConnectivity::cvih_ccwend >> ConstVertexIHalfedgeCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexOHalfedgeCCWIter, VertexHandle  , HalfedgeHandle, &PolyConnectivity::cvoh_ccwbegin , &PolyConnectivity::cvoh_ccwend >> ConstVertexOHalfedgeCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexEdgeCCWIter     , VertexHandle  , EdgeHandle    , &PolyConnectivity::cve_ccwbegin  , &PolyConnectivity::cve_ccwend  >> ConstVertexEdgeCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstVertexFaceCCWIter     , VertexHandle  , FaceHandle    , &PolyConnectivity::cvf_ccwbegin  , &PolyConnectivity::cvf_ccwend  >> ConstVertexFaceCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceVertexCCWIter     , FaceHandle    , VertexHandle  , &PolyConnectivity::cfv_ccwbegin  , &PolyConnectivity::cfv_ccwend  >> ConstFaceVertexCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceHalfedgeCCWIter   , FaceHandle    , HalfedgeHandle, &PolyConnectivity::cfh_ccwbegin  , &PolyConnectivity::cfh_ccwend  >> ConstFaceHalfedgeCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceEdgeCCWIter       , FaceHandle    , EdgeHandle    , &PolyConnectivity::cfe_ccwbegin  , &PolyConnectivity::cfe_ccwend  >> ConstFaceEdgeCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstFaceFaceCCWIter       , FaceHandle    , FaceHandle    , &PolyConnectivity::cff_ccwbegin  , &PolyConnectivity::cff_ccwend  >> ConstFaceFaceCCWRange;
  typedef CirculatorRange<CirculatorRangeTraitT<PolyConnectivity, ConstHalfedgeLoopCCWIter   , HalfedgeHandle, HalfedgeHandle, &PolyConnectivity::chl_ccwbegin  , &PolyConnectivity::chl_ccwend  >> ConstHalfedgeLoopCCWRange;

  /**
   * @return The vertices adjacent to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexVertexRange vv_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexIHalfedgeRange vih_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like vih_range(VertexHandle _heh.to()) but starts iteration at _heh
   */
  ConstVertexIHalfedgeRange vih_range(HalfedgeHandle _heh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexOHalfedgeRange voh_range(VertexHandle _vh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like voh_range(VertexHandle _heh.from()) but starts iteration at _heh
   */
  ConstVertexOHalfedgeRange voh_range(HalfedgeHandle _heh) const;

  /**
   * @return The edges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexEdgeRange ve_range(VertexHandle _vh) const ;

  /**
   * @return The faces incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexFaceRange vf_range(VertexHandle _vh) const;

  /**
   * @return The vertices incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceVertexRange fv_range(FaceHandle _fh) const;

  /**
   * @return The halfedges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceHalfedgeRange fh_range(FaceHandle _fh) const;

  /**
   * @return The edges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceEdgeRange fe_range(FaceHandle _fh) const;

  /**
   * @return The faces adjacent to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceFaceRange ff_range(FaceHandle _fh) const;

  /**
   * @return The vertices incident to the specified edge
   * as a range object suitable for C++11 range based for loops.
   */
  ConstEdgeVertexRange ev_range(EdgeHandle _eh) const;

  /**
   * @return The halfedges of the specified edge
   * as a range object suitable for C++11 range based for loops.
   */
  ConstEdgeHalfedgeRange eh_range(EdgeHandle _eh) const;

  /**
   * @return The halfedges of the specified edge
   * as a range object suitable for C++11 range based for loops.
   * Like eh_range(_heh.edge()) but starts iteration at _heh
   */
  ConstEdgeHalfedgeRange eh_range(HalfedgeHandle _heh) const;

  /**
   * @return The faces incident to the specified edge
   * as a range object suitable for C++11 range based for loops.
   */
  ConstEdgeFaceRange ef_range(EdgeHandle _eh) const;

  /**
   * @return The halfedges in the face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstHalfedgeLoopRange hl_range(HalfedgeHandle _heh) const;


  /**
   * @return The vertices adjacent to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexVertexCWRange vv_cw_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexIHalfedgeCWRange vih_cw_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like vih_cw_range(VertexHandle _heh.to()) but starts iteration at _heh
   */
  ConstVertexIHalfedgeCWRange vih_cw_range(HalfedgeHandle _heh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexOHalfedgeCWRange voh_cw_range(VertexHandle _vh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like voh_cw_range(VertexHandle _heh.from()) but starts iteration at _heh
   */
  ConstVertexOHalfedgeCWRange voh_cw_range(HalfedgeHandle _heh) const;

  /**
   * @return The edges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexEdgeCWRange ve_cw_range(VertexHandle _vh) const;

  /**
   * @return The faces incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexFaceCWRange vf_cw_range(VertexHandle _vh) const;

  /**
   * @return The vertices incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceVertexCWRange fv_cw_range(FaceHandle _fh) const;

  /**
   * @return The halfedges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceHalfedgeCWRange fh_cw_range(FaceHandle _fh) const;

  /**
   * @return The edges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceEdgeCWRange fe_cw_range(FaceHandle _fh) const;

  /**
   * @return The faces adjacent to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceFaceCWRange ff_cw_range(FaceHandle _fh) const;

  /**
   * @return The halfedges in the face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstHalfedgeLoopCWRange hl_cw_range(HalfedgeHandle _heh) const;


  /**
   * @return The vertices adjacent to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexVertexCCWRange vv_ccw_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexIHalfedgeCCWRange vih_ccw_range(VertexHandle _vh) const;

  /**
   * @return The incoming halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like vih_ccw_range(VertexHandle _heh.to()) but starts iteration at _heh
   */
  ConstVertexIHalfedgeCCWRange vih_ccw_range(HalfedgeHandle _heh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexOHalfedgeCCWRange voh_ccw_range(VertexHandle _vh) const;

  /**
   * @return The outgoing halfedges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   * Like voh_ccw_range(VertexHandle _heh.from()) but starts iteration at _heh
   */
  ConstVertexOHalfedgeCCWRange voh_ccw_range(HalfedgeHandle _heh) const;

  /**
   * @return The edges incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexEdgeCCWRange ve_ccw_range(VertexHandle _vh) const ;

  /**
   * @return The faces incident to the specified vertex
   * as a range object suitable for C++11 range based for loops.
   */
  ConstVertexFaceCCWRange vf_ccw_range(VertexHandle _vh) const;

  /**
   * @return The vertices incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceVertexCCWRange fv_ccw_range(FaceHandle _fh) const;

  /**
   * @return The halfedges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceHalfedgeCCWRange fh_ccw_range(FaceHandle _fh) const;

  /**
   * @return The edges incident to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceEdgeCCWRange fe_ccw_range(FaceHandle _fh) const;

  /**
   * @return The faces adjacent to the specified face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstFaceFaceCCWRange ff_ccw_range(FaceHandle _fh) const;


  /**
   * @return The halfedges in the face
   * as a range object suitable for C++11 range based for loops.
   */
  ConstHalfedgeLoopCCWRange hl_ccw_range(HalfedgeHandle _heh) const;

  //@}

  //===========================================================================
  /** @name Boundary and manifold tests
   * @{ */
  //===========================================================================

  /** \brief Check if the halfedge is at the boundary
   *
   * The halfedge is at the boundary, if no face is incident to it.
   *
   * @param _heh HalfedgeHandle to test
   * @return boundary?
   */
  bool is_boundary(HalfedgeHandle _heh) const
  { return ArrayKernel::is_boundary(_heh); }

  /** \brief Is the edge a boundary edge?
   *
   * Checks it the edge _eh is a boundary edge, i.e. is one of its halfedges
   * a boundary halfedge.
   *
   * @param _eh Edge handle to test
   * @return boundary?
   */
  bool is_boundary(EdgeHandle _eh) const
  {
    return (is_boundary(halfedge_handle(_eh, 0)) ||
            is_boundary(halfedge_handle(_eh, 1)));
  }

  /** \brief Is vertex _vh a boundary vertex ?
   *
   * Checks if the associated halfedge (which would on a boundary be the outside
   * halfedge), is connected to a face. Which is equivalent, if the vertex is
   * at the boundary of the mesh, as OpenMesh will make sure, that if there is a
   * boundary halfedge at the vertex, the halfedge will be the one which is associated
   * to the vertex.
   *
   * @param _vh VertexHandle to test
   * @return boundary?
   */
  bool is_boundary(VertexHandle _vh) const
  {
    HalfedgeHandle heh(halfedge_handle(_vh));
    return (!(heh.is_valid() && face_handle(heh).is_valid()));
  }

  /** \brief Check if face is at the boundary
   *
   * Is face _fh at boundary, i.e. is one of its edges (or vertices)
   * a boundary edge?
   *
   * @param _fh Check this face
   * @param _check_vertex If \c true, check the corner vertices of the face, too.
   * @return boundary?
   */
  bool is_boundary(FaceHandle _fh, bool _check_vertex=false) const;

  /** \brief Is (the mesh at) vertex _vh  two-manifold ?
   *
   * The vertex is non-manifold if more than one gap exists, i.e.
   * more than one outgoing boundary halfedge. If (at least) one
   * boundary halfedge exists, the vertex' halfedge must be a
   * boundary halfedge.
   *
   * @param _vh VertexHandle to test
   * @return manifold?
   */
  bool is_manifold(VertexHandle _vh) const;

  /** @} */
    
  // --- misc ---

  /** Adjust outgoing halfedge handle for vertices, so that it is a
      boundary halfedge whenever possible. 
  */
  void adjust_outgoing_halfedge(VertexHandle _vh);

  /// Find halfedge from _vh0 to _vh1. Returns invalid handle if not found.
  SmartHalfedgeHandle find_halfedge(VertexHandle _start_vh, VertexHandle _end_vh) const;
  /// Vertex valence
  uint valence(VertexHandle _vh) const;
  /// Face valence
  uint valence(FaceHandle _fh) const;
  
  // --- connectivity operattions 
    
  /** Halfedge collapse: collapse the from-vertex of halfedge _heh
      into its to-vertex.

      \attention Needs vertex/edge/face status attribute in order to
      delete the items that degenerate.

      \note The from vertex is marked as deleted while the to vertex will still exist.

      \note This function does not perform a garbage collection. It
      only marks degenerate items as deleted.

      \attention A halfedge collapse may lead to topological inconsistencies.
      Therefore you should check this using is_collapse_ok().  
  */
  void collapse(HalfedgeHandle _heh);
  /** return true if the this the only link between the faces adjacent to _eh.
      _eh is allowed to be boundary, in which case true is returned iff _eh is 
      the only boundary edge of its ajdacent face.
  */
  bool is_simple_link(EdgeHandle _eh) const;
  /** return true if _fh shares only one edge with all of its adjacent faces.
      Boundary is treated as one face, i.e., the function false if _fh has more
      than one boundary edge.
  */
  bool is_simply_connected(FaceHandle _fh) const;
  /** Removes the edge _eh. Its adjacent faces are merged. _eh and one of the 
      adjacent faces are set deleted. The handle of the remaining face is 
      returned (InvalidFaceHandle is returned if _eh is a boundary edge).
      
      \pre is_simple_link(_eh). This ensures that there are no hole faces
      or isolated vertices appearing in result of the operation.
      
      \attention Needs the Attributes::Status attribute for edges and faces.
      
      \note This function does not perform a garbage collection. It
      only marks items as deleted.
  */
  FaceHandle remove_edge(EdgeHandle _eh);
  /** Inverse of remove_edge. _eh should be the handle of the edge and the
      vertex and halfedge handles pointed by edge(_eh) should be valid. 
  */
  void reinsert_edge(EdgeHandle _eh);
  /** Inserts an edge between to_vh(_prev_heh) and from_vh(_next_heh).
      A new face is created started at heh0 of the inserted edge and
      its halfedges loop includes both _prev_heh and _next_heh. If an 
      old face existed which includes the argument halfedges, it is 
      split at the new edge. heh0 is returned. 
      
      \note assumes _prev_heh and _next_heh are either boundary or pointed
      to the same face
  */
  HalfedgeHandle insert_edge(HalfedgeHandle _prev_heh, HalfedgeHandle _next_heh);
    
  /** \brief Face split (= 1-to-n split).
     *
     * Split an arbitrary face into triangles by connecting each vertex of fh to vh.
     *
     * \note fh will remain valid (it will become one of the triangles)
     * \note the halfedge handles of the new triangles will point to the old halfeges
     *
     * \note The properties of the new faces and all other new primitives will be undefined!
     *
     * @param _fh Face handle that should be splitted
     * @param _vh Vertex handle of the new vertex that will be inserted in the face
     */
  void split(FaceHandle _fh, VertexHandle _vh);

  /** \brief Face split (= 1-to-n split).
   *
   * Split an arbitrary face into triangles by connecting each vertex of fh to vh.
   *
   * \note fh will remain valid (it will become one of the triangles)
   * \note the halfedge handles of the new triangles will point to the old halfeges
   *
   * \note The properties of the new faces will be adjusted to the properties of the original faces
   * \note Properties of the new edges and halfedges will be undefined
   *
   * @param _fh Face handle that should be splitted
   * @param _vh Vertex handle of the new vertex that will be inserted in the face
   */
  void split_copy(FaceHandle _fh, VertexHandle _vh);
  
  /** \brief Triangulate the face _fh

    Split an arbitrary face into triangles by connecting
    each vertex of fh after its second to vh.

    \note _fh will remain valid (it will become one of the
      triangles)

    \note The halfedge handles of the new triangles will
      point to the old halfedges

    @param _fh Handle of the face that should be triangulated
  */
  void triangulate(FaceHandle _fh);

  /** \brief triangulate the entire mesh  
  */
  void triangulate();
  
  /** Edge split (inserts a vertex on the edge only)
   *
   * This edge split only splits the edge without introducing new faces!
   * As this is for polygonal meshes, we can have valence 2 vertices here.
   *
   * \note The properties of the new edges and halfedges will be undefined!
   *
   * @param _eh Handle of the edge, that will be splitted
   * @param _vh Handle of the vertex that will be inserted at the edge
   */
  void split_edge(EdgeHandle _eh, VertexHandle _vh);

  /** Edge split (inserts a vertex on the edge only)
   *
   * This edge split only splits the edge without introducing new faces!
   * As this is for polygonal meshes, we can have valence 2 vertices here.
   *
   * \note The properties of the new edge will be copied from the splitted edge
   * \note Properties of the new halfedges will be undefined
   *
   * @param _eh Handle of the edge, that will be splitted
   * @param _vh Handle of the vertex that will be inserted at the edge
   */
  void split_edge_copy(EdgeHandle _eh, VertexHandle _vh);


  /** \name Generic handle derefertiation.
      Calls the respective vertex(), halfedge(), edge(), face()
      method of the mesh kernel.
  */
  //@{
  /// Get item from handle
  const Vertex&    deref(VertexHandle _h)   const { return vertex(_h); }
  Vertex&          deref(VertexHandle _h)         { return vertex(_h); }
  const Halfedge&  deref(HalfedgeHandle _h) const { return halfedge(_h); }
  Halfedge&        deref(HalfedgeHandle _h)       { return halfedge(_h); }
  const Edge&      deref(EdgeHandle _h)     const { return edge(_h); }
  Edge&            deref(EdgeHandle _h)           { return edge(_h); }
  const Face&      deref(FaceHandle _h)     const { return face(_h); }
  Face&            deref(FaceHandle _h)           { return face(_h); }
  //@}

protected:  
  /// Helper for halfedge collapse
  void collapse_edge(HalfedgeHandle _hh);
  /// Helper for halfedge collapse
  void collapse_loop(HalfedgeHandle _hh);



private: // Working storage for add_face()
       struct AddFaceEdgeInfo
       {
               HalfedgeHandle halfedge_handle;
               bool is_new;
               bool needs_adjust;
       };
       std::vector<AddFaceEdgeInfo> edgeData_; //
       std::vector<std::pair<HalfedgeHandle, HalfedgeHandle> > next_cache_; // cache for set_next_halfedge and vertex' set_halfedge

};

template <>
struct PolyConnectivity::ElementRange<VertexHandle>
{
  using Range         = ConstVertexRange;
  using RangeSkipping = ConstVertexRangeSkipping;
};

template <>
struct PolyConnectivity::ElementRange<HalfedgeHandle>
{
  using Range         = ConstHalfedgeRange;
  using RangeSkipping = ConstHalfedgeRangeSkipping;
};

template <>
struct PolyConnectivity::ElementRange<EdgeHandle>
{
  using Range         = ConstEdgeRange;
  using RangeSkipping = ConstEdgeRangeSkipping;
};

template <>
struct PolyConnectivity::ElementRange<FaceHandle>
{
  using Range         = ConstFaceRange;
  using RangeSkipping = ConstFaceRangeSkipping;
};

}//namespace OpenMesh

#define OPENMESH_POLYCONNECTIVITY_INTERFACE_INCLUDE
#include <OpenMesh/Core/Mesh/SmartHandles.hh>
#include <OpenMesh/Core/Mesh/PolyConnectivity_inline_impl.hh>
#undef OPENMESH_POLYCONNECTIVITY_INTERFACE_INCLUDE

#endif//OPENMESH_POLYCONNECTIVITY_HH
