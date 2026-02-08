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

#ifndef OPENMESH_POLYCONNECTIVITY_INTERFACE_INCLUDE
#error Do not include this directly, include instead PolyConnectivity.hh
#endif//OPENMESH_POLYCONNECTIVITY_INTERFACE_INCLUDE

#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>

//== NAMESPACES ===============================================================

namespace OpenMesh {

//== FORWARD DECLARATION ======================================================

struct SmartVertexHandle;
struct SmartHalfedgeHandle;
struct SmartEdgeHandle;
struct SmartFaceHandle;


//== CLASS DEFINITION =========================================================

/// Base class for all smart handle types
class OPENMESHDLLEXPORT SmartBaseHandle
{
public:
  explicit SmartBaseHandle(const PolyConnectivity* _mesh = nullptr) : mesh_(_mesh) {}

  /// Get the underlying mesh of this handle
  const PolyConnectivity* mesh() const { return mesh_; }

  // TODO: should operators ==, !=, < look at mesh_?

private:
  const PolyConnectivity* mesh_;

};

/// Base class for all smart handle types that contains status related methods
template <typename HandleType>
class SmartHandleStatusPredicates
{
public:
  /// Returns true iff the handle is marked as feature
  bool feature() const;
  /// Returns true iff the handle is marked as selected
  bool selected() const;
  /// Returns true iff the handle is marked as tagged
  bool tagged() const;
  /// Returns true iff the handle is marked as tagged2
  bool tagged2() const;
  /// Returns true iff the handle is marked as locked
  bool locked() const;
  /// Returns true iff the handle is marked as hidden
  bool hidden() const;
  /// Returns true iff the handle is marked as deleted
  bool deleted() const;
};

/// Base class for all smart handle types that contains status related methods
template <typename HandleType>
class SmartHandleBoundaryPredicate
{
public:
  /// Returns true iff the handle is boundary
  bool is_boundary() const;
};

/// Smart version of VertexHandle contains a pointer to the corresponding mesh and allows easier access to navigation methods
struct OPENMESHDLLEXPORT SmartVertexHandle : public SmartBaseHandle, VertexHandle, SmartHandleStatusPredicates<SmartVertexHandle>, SmartHandleBoundaryPredicate<SmartVertexHandle>
{
  explicit SmartVertexHandle(int _idx=-1, const PolyConnectivity* _mesh = nullptr) : SmartBaseHandle(_mesh), VertexHandle(_idx) {}

  /// Returns an outgoing halfedge
  SmartHalfedgeHandle out()      const;
  /// Returns an outgoing halfedge
  SmartHalfedgeHandle halfedge() const; // alias for out
  /// Returns an incoming halfedge
  SmartHalfedgeHandle in()       const;

  /// Returns a range of faces incident to the vertex (PolyConnectivity::vf_range())
  PolyConnectivity::ConstVertexFaceRange         faces()              const;
  /// Returns a range of faces incident to the vertex (PolyConnectivity::vf_cw_range())
  PolyConnectivity::ConstVertexFaceCWRange       faces_cw()           const;
  /// Returns a range of faces incident to the vertex (PolyConnectivity::vf_ccw_range())
  PolyConnectivity::ConstVertexFaceCCWRange      faces_ccw()          const;
  /// Returns a range of edges incident to the vertex (PolyConnectivity::ve_range())
  PolyConnectivity::ConstVertexEdgeRange         edges()              const;
  /// Returns a range of edges incident to the vertex (PolyConnectivity::ve_cw_range())
  PolyConnectivity::ConstVertexEdgeCWRange       edges_cw()           const;
  /// Returns a range of edges incident to the vertex (PolyConnectivity::ve_ccw_range())
  PolyConnectivity::ConstVertexEdgeCCWRange      edges_ccw()          const;
  /// Returns a range of vertices adjacent to the vertex (PolyConnectivity::vv_range())
  PolyConnectivity::ConstVertexVertexRange       vertices()           const;
  /// Returns a range of vertices adjacent to the vertex (PolyConnectivity::vv_cw_range())
  PolyConnectivity::ConstVertexVertexCWRange     vertices_cw()           const;
  /// Returns a range of vertices adjacent to the vertex (PolyConnectivity::vv_ccw_range())
  PolyConnectivity::ConstVertexVertexCCWRange    vertices_ccw()           const;
  /// Returns a range of outgoing halfedges incident to the vertex (PolyConnectivity::voh_range())
  PolyConnectivity::ConstVertexIHalfedgeRange    incoming_halfedges() const;
  /// Returns a range of outgoing halfedges incident to the vertex (PolyConnectivity::voh_cw_range())
  PolyConnectivity::ConstVertexIHalfedgeCWRange  incoming_halfedges_cw() const;
  /// Returns a range of outgoing halfedges incident to the vertex (PolyConnectivity::voh_ccw_range())
  PolyConnectivity::ConstVertexIHalfedgeCCWRange incoming_halfedges_ccw() const;
  /// Returns a range of outgoing halfedges incident to the vertex (PolyConnectivity::vih_range())
  PolyConnectivity::ConstVertexIHalfedgeRange    incoming_halfedges(HalfedgeHandle _heh) const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::vih_cw_range())
  PolyConnectivity::ConstVertexIHalfedgeCWRange  incoming_halfedges_cw(HalfedgeHandle _heh) const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::vih_ccw_range())
  PolyConnectivity::ConstVertexIHalfedgeCCWRange incoming_halfedges_ccw(HalfedgeHandle _heh) const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_range())
  PolyConnectivity::ConstVertexOHalfedgeRange    outgoing_halfedges() const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_cw_range())
  PolyConnectivity::ConstVertexOHalfedgeCWRange  outgoing_halfedges_cw() const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_ccw_range())
  PolyConnectivity::ConstVertexOHalfedgeCCWRange outgoing_halfedges_ccw() const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_range())
  PolyConnectivity::ConstVertexOHalfedgeRange    outgoing_halfedges(HalfedgeHandle _heh) const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_cw_range())
  PolyConnectivity::ConstVertexOHalfedgeCWRange  outgoing_halfedges_cw(HalfedgeHandle _heh) const;
  /// Returns a range of incoming halfedges incident to the vertex (PolyConnectivity::voh_ccw_range())
  PolyConnectivity::ConstVertexOHalfedgeCCWRange outgoing_halfedges_ccw(HalfedgeHandle _heh) const;

  /// Returns valence of the vertex
  uint valence()     const;
  /// Returns true iff (the mesh at) the vertex is two-manifold ?
  bool is_manifold() const;
};

struct OPENMESHDLLEXPORT SmartHalfedgeHandle : public SmartBaseHandle, HalfedgeHandle, SmartHandleStatusPredicates<SmartHalfedgeHandle>, SmartHandleBoundaryPredicate<SmartHalfedgeHandle>
{
  explicit SmartHalfedgeHandle(int _idx=-1, const PolyConnectivity* _mesh = nullptr) : SmartBaseHandle(_mesh), HalfedgeHandle(_idx) {}

  /// Returns next halfedge handle
  SmartHalfedgeHandle next() const;
  /// Returns previous halfedge handle
  SmartHalfedgeHandle prev() const;
  /// Returns opposite halfedge handle
  SmartHalfedgeHandle opp()  const;
  /// Returns vertex pointed to by halfedge
  SmartVertexHandle   to()   const;
  /// Returns vertex at start of halfedge
  SmartVertexHandle   from() const;
  /// Returns incident edge of halfedge
  SmartEdgeHandle     edge() const;
  /// Returns incident face of halfedge
  SmartFaceHandle     face() const;

  /// Returns a range of halfedges in the face of the halfedge (or along the boundary) (PolyConnectivity::hl_range())
  PolyConnectivity::ConstHalfedgeLoopRange    loop()     const;
  /// Returns a range of halfedges in the face of the halfedge (or along the boundary) (PolyConnectivity::hl_cw_range())
  PolyConnectivity::ConstHalfedgeLoopCWRange  loop_cw()  const;
  /// Returns a range of halfedges in the face of the halfedge (or along the boundary) (PolyConnectivity::hl_ccw_range())
  PolyConnectivity::ConstHalfedgeLoopCCWRange loop_ccw() const;
};

struct OPENMESHDLLEXPORT SmartEdgeHandle : public SmartBaseHandle, EdgeHandle, SmartHandleStatusPredicates<SmartEdgeHandle>, SmartHandleBoundaryPredicate<SmartEdgeHandle>
{
  explicit SmartEdgeHandle(int _idx=-1, const PolyConnectivity* _mesh = nullptr) : SmartBaseHandle(_mesh), EdgeHandle(_idx) {}

  /// Returns one of the two halfedges of the edge
  SmartHalfedgeHandle halfedge(unsigned int _i) const;
  /// Shorthand for halfedge()
  SmartHalfedgeHandle h(unsigned int _i)        const;
  /// Shorthand for halfedge(0)
  SmartHalfedgeHandle h0()                      const;
  /// Shorthand for halfedge(1)
  SmartHalfedgeHandle h1()                      const;
  /// Returns one of the two incident vertices of the edge
  SmartVertexHandle   vertex(unsigned int _i)   const;
  /// Shorthand for vertex()
  SmartVertexHandle   v(unsigned int _i)        const;
  /// Shorthand for vertex(0)
  SmartVertexHandle   v0()                      const;
  /// Shorthand for vertex(1)
  SmartVertexHandle   v1()                      const;

  /// Returns a range of vertices incident to the edge (PolyConnectivity::ev_range())
  PolyConnectivity::ConstEdgeVertexRange   vertices()  const;
  /// Returns a range of halfedges of the edge (PolyConnectivity::eh_range())
  PolyConnectivity::ConstEdgeHalfedgeRange halfedges() const;
  /// Returns a range of halfedges of the edge (PolyConnectivity::eh_range())
  PolyConnectivity::ConstEdgeHalfedgeRange halfedges(HalfedgeHandle _heh) const;
  /// Returns a range of faces incident to the edge (PolyConnectivity::ef_range())
  PolyConnectivity::ConstEdgeFaceRange     faces()     const;
};

struct OPENMESHDLLEXPORT SmartFaceHandle : public SmartBaseHandle, FaceHandle, SmartHandleStatusPredicates<SmartFaceHandle>, SmartHandleBoundaryPredicate<SmartFaceHandle>
{
  explicit SmartFaceHandle(int _idx=-1, const PolyConnectivity* _mesh = nullptr) : SmartBaseHandle(_mesh), FaceHandle(_idx) {}

  /// Returns one of the halfedges of the face
  SmartHalfedgeHandle halfedge() const;

  /// Returns a range of vertices incident to the face (PolyConnectivity::fv_range())
  PolyConnectivity::ConstFaceVertexRange      vertices()      const;
  /// Returns a range of vertices incident to the face (PolyConnectivity::fv_cw_range())
  PolyConnectivity::ConstFaceVertexCWRange    vertices_cw()   const;
  /// Returns a range of vertices incident to the face (PolyConnectivity::fv_ccw_range())
  PolyConnectivity::ConstFaceVertexCCWRange   vertices_ccw()  const;
  /// Returns a range of halfedges of the face (PolyConnectivity::fh_range())
  PolyConnectivity::ConstFaceHalfedgeRange    halfedges()     const;
  /// Returns a range of halfedges of the face (PolyConnectivity::fh_cw_range())
  PolyConnectivity::ConstFaceHalfedgeCWRange  halfedges_cw()  const;
  /// Returns a range of halfedges of the face (PolyConnectivity::fh_ccw_range())
  PolyConnectivity::ConstFaceHalfedgeCCWRange halfedges_ccw() const;
  /// Returns a range of edges of the face (PolyConnectivity::fv_range())
  PolyConnectivity::ConstFaceEdgeRange        edges()         const;
  /// Returns a range of edges of the face (PolyConnectivity::fv_cw_range())
  PolyConnectivity::ConstFaceEdgeCWRange      edges_cw()      const;
  /// Returns a range of edges of the face (PolyConnectivity::fv_ccw_range())
  PolyConnectivity::ConstFaceEdgeCCWRange     edges_ccw()     const;
  /// Returns a range adjacent faces of the face (PolyConnectivity::ff_range())
  PolyConnectivity::ConstFaceFaceRange        faces()         const;
  /// Returns a range adjacent faces of the face (PolyConnectivity::ff_cw_range())
  PolyConnectivity::ConstFaceFaceCWRange      faces_cw()      const;
  /// Returns a range adjacent faces of the face (PolyConnectivity::ff_ccw_range())
  PolyConnectivity::ConstFaceFaceCCWRange     faces_ccw()     const;

  /// Returns the valence of the face
  uint valence()     const;
};


/// Creats a SmartVertexHandle from a VertexHandle and a Mesh
inline SmartVertexHandle   make_smart(VertexHandle   _vh, const PolyConnectivity* _mesh) { return SmartVertexHandle  (_vh.idx(), _mesh); }
/// Creats a SmartHalfedgeHandle from a HalfedgeHandle and a Mesh
inline SmartHalfedgeHandle make_smart(HalfedgeHandle _hh, const PolyConnectivity* _mesh) { return SmartHalfedgeHandle(_hh.idx(), _mesh); }
/// Creats a SmartEdgeHandle from an EdgeHandle and a Mesh
inline SmartEdgeHandle     make_smart(EdgeHandle     _eh, const PolyConnectivity* _mesh) { return SmartEdgeHandle    (_eh.idx(), _mesh); }
/// Creats a SmartFaceHandle from a FaceHandle and a Mesh
inline SmartFaceHandle     make_smart(FaceHandle     _fh, const PolyConnectivity* _mesh) { return SmartFaceHandle    (_fh.idx(), _mesh); }

/// Creats a SmartVertexHandle from a VertexHandle and a Mesh
inline SmartVertexHandle   make_smart(VertexHandle   _vh, const PolyConnectivity& _mesh) { return SmartVertexHandle  (_vh.idx(), &_mesh); }
/// Creats a SmartHalfedgeHandle from a HalfedgeHandle and a Mesh
inline SmartHalfedgeHandle make_smart(HalfedgeHandle _hh, const PolyConnectivity& _mesh) { return SmartHalfedgeHandle(_hh.idx(), &_mesh); }
/// Creats a SmartEdgeHandle from an EdgeHandle and a Mesh
inline SmartEdgeHandle     make_smart(EdgeHandle     _eh, const PolyConnectivity& _mesh) { return SmartEdgeHandle    (_eh.idx(), &_mesh); }
/// Creats a SmartFaceHandle from a FaceHandle and a Mesh
inline SmartFaceHandle     make_smart(FaceHandle     _fh, const PolyConnectivity& _mesh) { return SmartFaceHandle    (_fh.idx(), &_mesh); }


// helper to convert Handle Types to Smarthandle Types
template <typename HandleT>
struct SmartHandle;

template <> struct SmartHandle<VertexHandle>   { using type = SmartVertexHandle;   };
template <> struct SmartHandle<HalfedgeHandle> { using type = SmartHalfedgeHandle; };
template <> struct SmartHandle<EdgeHandle>     { using type = SmartEdgeHandle;     };
template <> struct SmartHandle<FaceHandle>     { using type = SmartFaceHandle;     };


template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::feature() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).feature();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::selected() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).selected();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::tagged() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).tagged();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::tagged2() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).tagged2();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::locked() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).locked();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::hidden() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).hidden();
}

template <typename HandleType>
inline bool SmartHandleStatusPredicates<HandleType>::deleted() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->status(handle).deleted();
}

template <typename HandleType>
inline bool SmartHandleBoundaryPredicate<HandleType>::is_boundary() const
{
  const auto& handle = static_cast<const HandleType&>(*this);
  assert(handle.mesh() != nullptr);
  return handle.mesh()->is_boundary(handle);
}

inline SmartHalfedgeHandle SmartVertexHandle::out() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->halfedge_handle(*this), mesh());
}

inline SmartHalfedgeHandle SmartVertexHandle::halfedge() const
{
  return out();
}

inline SmartHalfedgeHandle SmartVertexHandle::in() const
{
  return out().opp();
}

inline uint SmartVertexHandle::valence() const
{
  assert(mesh() != nullptr);
  return mesh()->valence(*this);
}

inline bool SmartVertexHandle::is_manifold() const
{
  assert(mesh() != nullptr);
  return mesh()->is_manifold(*this);
}

inline SmartHalfedgeHandle SmartHalfedgeHandle::next() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->next_halfedge_handle(*this), mesh());
}

inline SmartHalfedgeHandle SmartHalfedgeHandle::prev() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->prev_halfedge_handle(*this), mesh());
}

inline SmartHalfedgeHandle SmartHalfedgeHandle::opp() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->opposite_halfedge_handle(*this), mesh());
}

inline SmartVertexHandle SmartHalfedgeHandle::to() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->to_vertex_handle(*this), mesh());
}

inline SmartVertexHandle SmartHalfedgeHandle::from() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->from_vertex_handle(*this), mesh());
}

inline SmartEdgeHandle SmartHalfedgeHandle::edge() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->edge_handle(*this), mesh());
}

inline SmartFaceHandle SmartHalfedgeHandle::face() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->face_handle(*this), mesh());
}

inline SmartHalfedgeHandle SmartEdgeHandle::halfedge(unsigned int _i = 0) const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->halfedge_handle(*this, _i), mesh());
}

inline SmartHalfedgeHandle SmartEdgeHandle::h(unsigned int _i = 0) const
{
  return halfedge(_i);
}

inline SmartHalfedgeHandle SmartEdgeHandle::h0() const
{
  return h(0);
}

inline SmartHalfedgeHandle SmartEdgeHandle::h1() const
{
  return h(1);
}

inline SmartVertexHandle SmartEdgeHandle::vertex(unsigned int _i) const
{
  return halfedge(_i).from();
}

inline SmartVertexHandle SmartEdgeHandle::v(unsigned int _i) const
{
  return vertex(_i);
}

inline SmartVertexHandle SmartEdgeHandle::v0() const
{
  return v(0);
}

inline SmartVertexHandle SmartEdgeHandle::v1() const
{
  return v(1);
}

inline SmartHalfedgeHandle SmartFaceHandle::halfedge() const
{
  assert(mesh() != nullptr);
  return make_smart(mesh()->halfedge_handle(*this), mesh());
}

inline uint SmartFaceHandle::valence() const
{
  assert(mesh() != nullptr);
  return mesh()->valence(*this);
}

//=============================================================================
} // namespace OpenMesh
//=============================================================================

//=============================================================================
