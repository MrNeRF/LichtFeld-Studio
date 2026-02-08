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
#endif // OPENMESH_POLYCONNECTIVITY_INTERFACE_INCLUDE

#include <OpenMesh/Core/Mesh/PolyConnectivity.hh> // To help some IDEs
#include <OpenMesh/Core/Mesh/IteratorsT.hh>
#include <OpenMesh/Core/Mesh/CirculatorsT.hh>

namespace OpenMesh {


inline SmartVertexHandle PolyConnectivity::add_vertex() { return make_smart(new_vertex(), *this); }

inline SmartHalfedgeHandle PolyConnectivity::next_halfedge_handle(SmartHalfedgeHandle _heh) const        { return make_smart(next_halfedge_handle(HalfedgeHandle(_heh)),        *this); }
inline SmartHalfedgeHandle PolyConnectivity::prev_halfedge_handle(SmartHalfedgeHandle _heh) const        { return make_smart(prev_halfedge_handle(HalfedgeHandle(_heh)),        *this); }
inline SmartHalfedgeHandle PolyConnectivity::opposite_halfedge_handle(SmartHalfedgeHandle _heh) const    { return make_smart(opposite_halfedge_handle(HalfedgeHandle(_heh)),    *this); }
inline SmartHalfedgeHandle PolyConnectivity::ccw_rotated_halfedge_handle(SmartHalfedgeHandle _heh) const { return make_smart(ccw_rotated_halfedge_handle(HalfedgeHandle(_heh)), *this); }
inline SmartHalfedgeHandle PolyConnectivity::cw_rotated_halfedge_handle(SmartHalfedgeHandle _heh) const  { return make_smart(cw_rotated_halfedge_handle(HalfedgeHandle(_heh)),  *this); }

inline SmartHalfedgeHandle PolyConnectivity::s_halfedge_handle(SmartEdgeHandle _eh, unsigned int _i)     { return make_smart(ArrayKernel::s_halfedge_handle(EdgeHandle(_eh), _i), _eh.mesh()); }
inline SmartEdgeHandle     PolyConnectivity::s_edge_handle(SmartHalfedgeHandle _heh)                     { return make_smart(ArrayKernel::s_edge_handle(HalfedgeHandle(_heh)), _heh.mesh()); }

inline SmartHalfedgeHandle PolyConnectivity::halfedge_handle(SmartEdgeHandle _eh, unsigned int _i) const { return make_smart(halfedge_handle(EdgeHandle(_eh), _i), *this); }
inline SmartEdgeHandle PolyConnectivity::edge_handle(SmartHalfedgeHandle _heh) const                     { return make_smart(edge_handle(HalfedgeHandle(_heh)),    *this); }
inline SmartHalfedgeHandle PolyConnectivity::halfedge_handle(SmartFaceHandle _fh) const                  { return make_smart(halfedge_handle(FaceHandle(_fh)),     *this); }
inline SmartHalfedgeHandle PolyConnectivity::halfedge_handle(SmartVertexHandle _vh) const                { return make_smart(halfedge_handle(VertexHandle(_vh)),   *this); }

inline SmartFaceHandle PolyConnectivity::face_handle(SmartHalfedgeHandle _heh) const                     { return make_smart(face_handle(HalfedgeHandle(_heh)),    *this); }

inline SmartFaceHandle PolyConnectivity::opposite_face_handle(HalfedgeHandle _heh) const                 { return make_smart(face_handle(opposite_halfedge_handle(_heh)), *this); }


/// Generic class for vertex/halfedge/edge/face ranges.
template <typename RangeTraitT>
class EntityRange : public SmartRangeT<EntityRange<RangeTraitT>, typename RangeTraitT::ITER_TYPE::SmartHandle> {
    public:
        typedef typename RangeTraitT::ITER_TYPE iterator;
        typedef typename RangeTraitT::ITER_TYPE const_iterator;

        explicit EntityRange(typename RangeTraitT::CONTAINER_TYPE &container) : container_(&container) {}
        typename RangeTraitT::ITER_TYPE begin() const { return RangeTraitT::begin(*container_); }
        typename RangeTraitT::ITER_TYPE end() const { return RangeTraitT::end(*container_); }

    private:
        typename RangeTraitT::CONTAINER_TYPE *container_;
};

/// Generic class for iterator ranges.
template <typename CirculatorRangeTraitT>
//class CirculatorRange : public SmartRangeT<CirculatorRange<CirculatorRangeTraitT>, decltype (make_smart(std::declval<typename CirculatorRangeTraitT::TO_ENTITYE_TYPE>(), std::declval<PolyConnectivity>()))>{
class CirculatorRange : public SmartRangeT<CirculatorRange<CirculatorRangeTraitT>, typename SmartHandle<typename CirculatorRangeTraitT::TO_ENTITYE_TYPE>::type>{
    public:
        typedef typename CirculatorRangeTraitT::ITER_TYPE ITER_TYPE;
        typedef typename CirculatorRangeTraitT::CENTER_ENTITY_TYPE CENTER_ENTITY_TYPE;
        typedef typename CirculatorRangeTraitT::CONTAINER_TYPE CONTAINER_TYPE;
        typedef ITER_TYPE iterator;
        typedef ITER_TYPE const_iterator;
        CirculatorRange(
                const CONTAINER_TYPE &container,
                CENTER_ENTITY_TYPE center) :
            container_(&container), heh_()
        {
          auto it = CirculatorRangeTraitT::begin(*container_, center);
          heh_ = it.heh_;
        }

        CirculatorRange(
                const CONTAINER_TYPE &container,
                HalfedgeHandle heh, int) :
            container_(&container), heh_(heh) {}

        ITER_TYPE begin() const { return CirculatorRangeTraitT::begin(*container_, heh_, 1); }
        ITER_TYPE end()   const { return CirculatorRangeTraitT::end(*container_, heh_, 1); }

    private:
        const CONTAINER_TYPE *container_;
        HalfedgeHandle heh_;
};


inline PolyConnectivity::ConstVertexRangeSkipping   PolyConnectivity::vertices()      const { return ConstVertexRangeSkipping(*this);   }
inline PolyConnectivity::ConstVertexRange           PolyConnectivity::all_vertices()  const { return ConstVertexRange(*this);           }
inline PolyConnectivity::ConstHalfedgeRangeSkipping PolyConnectivity::halfedges()     const { return ConstHalfedgeRangeSkipping(*this); }
inline PolyConnectivity::ConstHalfedgeRange         PolyConnectivity::all_halfedges() const { return ConstHalfedgeRange(*this);         }
inline PolyConnectivity::ConstEdgeRangeSkipping     PolyConnectivity::edges()         const { return ConstEdgeRangeSkipping(*this);     }
inline PolyConnectivity::ConstEdgeRange             PolyConnectivity::all_edges()     const { return ConstEdgeRange(*this);             }
inline PolyConnectivity::ConstFaceRangeSkipping     PolyConnectivity::faces()         const { return ConstFaceRangeSkipping(*this);     }
inline PolyConnectivity::ConstFaceRange             PolyConnectivity::all_faces()     const { return ConstFaceRange(*this);             }

template <> inline PolyConnectivity::ConstVertexRangeSkipping   PolyConnectivity::elements<VertexHandle>()       const { return vertices();      }
template <> inline PolyConnectivity::ConstVertexRange           PolyConnectivity::all_elements<VertexHandle>()   const { return all_vertices();  }
template <> inline PolyConnectivity::ConstHalfedgeRangeSkipping PolyConnectivity::elements<HalfedgeHandle>()     const { return halfedges();     }
template <> inline PolyConnectivity::ConstHalfedgeRange         PolyConnectivity::all_elements<HalfedgeHandle>() const { return all_halfedges(); }
template <> inline PolyConnectivity::ConstEdgeRangeSkipping     PolyConnectivity::elements<EdgeHandle>()         const { return edges();         }
template <> inline PolyConnectivity::ConstEdgeRange             PolyConnectivity::all_elements<EdgeHandle>()     const { return all_edges();     }
template <> inline PolyConnectivity::ConstFaceRangeSkipping     PolyConnectivity::elements<FaceHandle>()         const { return faces();         }
template <> inline PolyConnectivity::ConstFaceRange             PolyConnectivity::all_elements<FaceHandle>()     const { return all_faces();     }


inline PolyConnectivity::ConstVertexVertexRange PolyConnectivity::vv_range(VertexHandle _vh) const {
    return ConstVertexVertexRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeRange PolyConnectivity::vih_range(VertexHandle _vh) const {
    return ConstVertexIHalfedgeRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeRange PolyConnectivity::vih_range(HalfedgeHandle _heh) const {
    return ConstVertexIHalfedgeRange(*this, opposite_halfedge_handle(_heh), 1);
}

inline PolyConnectivity::ConstVertexOHalfedgeRange PolyConnectivity::voh_range(VertexHandle _vh) const {
    return ConstVertexOHalfedgeRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexOHalfedgeRange PolyConnectivity::voh_range(HalfedgeHandle _heh) const {
    return ConstVertexOHalfedgeRange(*this, _heh, 1);
}

inline PolyConnectivity::ConstVertexEdgeRange PolyConnectivity::ve_range(VertexHandle _vh) const {
    return ConstVertexEdgeRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexFaceRange PolyConnectivity::vf_range(VertexHandle _vh) const {
    return ConstVertexFaceRange(*this, _vh);
}

inline PolyConnectivity::ConstFaceVertexRange PolyConnectivity::fv_range(FaceHandle _fh) const {
    return ConstFaceVertexRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceHalfedgeRange PolyConnectivity::fh_range(FaceHandle _fh) const {
    return ConstFaceHalfedgeRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceEdgeRange PolyConnectivity::fe_range(FaceHandle _fh) const {
    return ConstFaceEdgeRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceFaceRange PolyConnectivity::ff_range(FaceHandle _fh) const {
    return ConstFaceFaceRange(*this, _fh);
}

inline PolyConnectivity::ConstEdgeVertexRange PolyConnectivity::ev_range(EdgeHandle _eh) const {
    return ConstEdgeVertexRange(*this, _eh);
}

inline PolyConnectivity::ConstEdgeHalfedgeRange PolyConnectivity::eh_range(EdgeHandle _eh) const {
    return ConstEdgeHalfedgeRange(*this, _eh);
}

inline PolyConnectivity::ConstEdgeHalfedgeRange PolyConnectivity::eh_range(HalfedgeHandle _heh) const {
    return ConstEdgeHalfedgeRange(*this, _heh, 1);
}

inline PolyConnectivity::ConstEdgeFaceRange PolyConnectivity::ef_range(EdgeHandle _eh) const {
    return ConstEdgeFaceRange(*this, _eh);
}

inline PolyConnectivity::ConstHalfedgeLoopRange PolyConnectivity::hl_range(HalfedgeHandle _heh) const {
    return ConstHalfedgeLoopRange(*this, _heh);
}


inline PolyConnectivity::ConstVertexVertexCWRange PolyConnectivity::vv_cw_range(VertexHandle _vh) const {
    return ConstVertexVertexCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeCWRange PolyConnectivity::vih_cw_range(VertexHandle _vh) const {
    return ConstVertexIHalfedgeCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeCWRange PolyConnectivity::vih_cw_range(HalfedgeHandle _heh) const {
    return ConstVertexIHalfedgeCWRange(*this, opposite_halfedge_handle(_heh), 1);
}

inline PolyConnectivity::ConstVertexOHalfedgeCWRange PolyConnectivity::voh_cw_range(VertexHandle _vh) const {
    return ConstVertexOHalfedgeCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexOHalfedgeCWRange PolyConnectivity::voh_cw_range(HalfedgeHandle _heh) const {
    return ConstVertexOHalfedgeCWRange(*this, _heh, 1);
}

inline PolyConnectivity::ConstVertexEdgeCWRange PolyConnectivity::ve_cw_range(VertexHandle _vh) const {
    return ConstVertexEdgeCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexFaceCWRange PolyConnectivity::vf_cw_range(VertexHandle _vh) const {
    return ConstVertexFaceCWRange(*this, _vh);
}

inline PolyConnectivity::ConstFaceVertexCWRange PolyConnectivity::fv_cw_range(FaceHandle _fh) const {
    return ConstFaceVertexCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceHalfedgeCWRange PolyConnectivity::fh_cw_range(FaceHandle _fh) const {
    return ConstFaceHalfedgeCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceEdgeCWRange PolyConnectivity::fe_cw_range(FaceHandle _fh) const {
    return ConstFaceEdgeCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceFaceCWRange PolyConnectivity::ff_cw_range(FaceHandle _fh) const {
    return ConstFaceFaceCWRange(*this, _fh);
}

inline PolyConnectivity::ConstHalfedgeLoopCWRange PolyConnectivity::hl_cw_range(HalfedgeHandle _heh) const {
    return ConstHalfedgeLoopCWRange(*this, _heh);
}



inline PolyConnectivity::ConstVertexVertexCCWRange PolyConnectivity::vv_ccw_range(VertexHandle _vh) const {
    return ConstVertexVertexCCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeCCWRange PolyConnectivity::vih_ccw_range(VertexHandle _vh) const {
    return ConstVertexIHalfedgeCCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexIHalfedgeCCWRange PolyConnectivity::vih_ccw_range(HalfedgeHandle _heh) const {
    return ConstVertexIHalfedgeCCWRange(*this, opposite_halfedge_handle(_heh), 1);
}

inline PolyConnectivity::ConstVertexOHalfedgeCCWRange PolyConnectivity::voh_ccw_range(VertexHandle _vh) const {
    return ConstVertexOHalfedgeCCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexOHalfedgeCCWRange PolyConnectivity::voh_ccw_range(HalfedgeHandle _heh) const {
    return ConstVertexOHalfedgeCCWRange(*this, _heh, 1);
}

inline PolyConnectivity::ConstVertexEdgeCCWRange PolyConnectivity::ve_ccw_range(VertexHandle _vh) const {
    return ConstVertexEdgeCCWRange(*this, _vh);
}

inline PolyConnectivity::ConstVertexFaceCCWRange PolyConnectivity::vf_ccw_range(VertexHandle _vh) const {
    return ConstVertexFaceCCWRange(*this, _vh);
}

inline PolyConnectivity::ConstFaceVertexCCWRange PolyConnectivity::fv_ccw_range(FaceHandle _fh) const {
    return ConstFaceVertexCCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceHalfedgeCCWRange PolyConnectivity::fh_ccw_range(FaceHandle _fh) const {
    return ConstFaceHalfedgeCCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceEdgeCCWRange PolyConnectivity::fe_ccw_range(FaceHandle _fh) const {
    return ConstFaceEdgeCCWRange(*this, _fh);
}

inline PolyConnectivity::ConstFaceFaceCCWRange PolyConnectivity::ff_ccw_range(FaceHandle _fh) const {
    return ConstFaceFaceCCWRange(*this, _fh);
}


inline PolyConnectivity::ConstHalfedgeLoopCCWRange PolyConnectivity::hl_ccw_range(HalfedgeHandle _heh) const {
    return ConstHalfedgeLoopCCWRange(*this, _heh);
}



inline PolyConnectivity::VertexIter PolyConnectivity::vertices_begin()
{  return VertexIter(*this, VertexHandle(0)); }

inline PolyConnectivity::ConstVertexIter PolyConnectivity::vertices_begin() const
{  return ConstVertexIter(*this, VertexHandle(0)); }

inline PolyConnectivity::VertexIter PolyConnectivity::vertices_end()
{  return VertexIter(*this, VertexHandle( int(n_vertices() ) )); }

inline PolyConnectivity::ConstVertexIter PolyConnectivity::vertices_end() const
{  return ConstVertexIter(*this, VertexHandle( int(n_vertices()) )); }

inline PolyConnectivity::HalfedgeIter PolyConnectivity::halfedges_begin()
{  return HalfedgeIter(*this, HalfedgeHandle(0)); }

inline PolyConnectivity::ConstHalfedgeIter PolyConnectivity::halfedges_begin() const
{  return ConstHalfedgeIter(*this, HalfedgeHandle(0)); }

inline PolyConnectivity::HalfedgeIter PolyConnectivity::halfedges_end()
{  return HalfedgeIter(*this, HalfedgeHandle(int(n_halfedges()))); }

inline PolyConnectivity::ConstHalfedgeIter PolyConnectivity::halfedges_end() const
{  return ConstHalfedgeIter(*this, HalfedgeHandle(int(n_halfedges()))); }

inline PolyConnectivity::EdgeIter PolyConnectivity::edges_begin()
{  return EdgeIter(*this, EdgeHandle(0)); }

inline PolyConnectivity::ConstEdgeIter PolyConnectivity::edges_begin() const
{  return ConstEdgeIter(*this, EdgeHandle(0)); }

inline PolyConnectivity::EdgeIter PolyConnectivity::edges_end()
{  return EdgeIter(*this, EdgeHandle(int(n_edges()))); }

inline PolyConnectivity::ConstEdgeIter PolyConnectivity::edges_end() const
{  return ConstEdgeIter(*this, EdgeHandle(int(n_edges()))); }

inline PolyConnectivity::FaceIter PolyConnectivity::faces_begin()
{  return FaceIter(*this, FaceHandle(0)); }

inline PolyConnectivity::ConstFaceIter PolyConnectivity::faces_begin() const
{  return ConstFaceIter(*this, FaceHandle(0)); }

inline PolyConnectivity::FaceIter PolyConnectivity::faces_end()
{  return FaceIter(*this, FaceHandle(int(n_faces()))); }


inline PolyConnectivity::ConstFaceIter PolyConnectivity::faces_end() const
{  return ConstFaceIter(*this, FaceHandle(int(n_faces()))); }

inline PolyConnectivity::VertexIter PolyConnectivity::vertices_sbegin()
{  return VertexIter(*this, VertexHandle(0), true); }

inline PolyConnectivity::ConstVertexIter PolyConnectivity::vertices_sbegin() const
{  return ConstVertexIter(*this, VertexHandle(0), true); }

inline PolyConnectivity::HalfedgeIter PolyConnectivity::halfedges_sbegin()
{  return HalfedgeIter(*this, HalfedgeHandle(0), true); }

inline PolyConnectivity::ConstHalfedgeIter PolyConnectivity::halfedges_sbegin() const
{  return ConstHalfedgeIter(*this, HalfedgeHandle(0), true); }

inline PolyConnectivity::EdgeIter PolyConnectivity::edges_sbegin()
{  return EdgeIter(*this, EdgeHandle(0), true); }

inline PolyConnectivity::ConstEdgeIter PolyConnectivity::edges_sbegin() const
{  return ConstEdgeIter(*this, EdgeHandle(0), true); }

inline PolyConnectivity::FaceIter PolyConnectivity::faces_sbegin()
{  return FaceIter(*this, FaceHandle(0), true); }

inline PolyConnectivity::ConstFaceIter PolyConnectivity::faces_sbegin() const
{  return ConstFaceIter(*this, FaceHandle(0), true); }

inline PolyConnectivity::VertexVertexIter PolyConnectivity::vv_iter(ArrayKernel::VertexHandle _vh)
{  return VertexVertexIter(*this, _vh); }

inline PolyConnectivity::VertexVertexCWIter PolyConnectivity::vv_cwiter(ArrayKernel::VertexHandle _vh)
{  return VertexVertexCWIter(*this, _vh); }

inline PolyConnectivity::VertexVertexCCWIter PolyConnectivity::vv_ccwiter(ArrayKernel::VertexHandle _vh)
{  return VertexVertexCCWIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeIter PolyConnectivity::vih_iter(ArrayKernel::VertexHandle _vh)
{  return VertexIHalfedgeIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeCWIter PolyConnectivity::vih_cwiter(ArrayKernel::VertexHandle _vh)
{  return VertexIHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeCCWIter PolyConnectivity::vih_ccwiter(ArrayKernel::VertexHandle _vh)
{  return VertexIHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeIter PolyConnectivity::voh_iter(ArrayKernel::VertexHandle _vh)
{  return VertexOHalfedgeIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeCWIter PolyConnectivity::voh_cwiter(ArrayKernel::VertexHandle _vh)
{  return VertexOHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeCCWIter PolyConnectivity::voh_ccwiter(ArrayKernel::VertexHandle _vh)
{  return VertexOHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeIter PolyConnectivity::ve_iter(ArrayKernel::VertexHandle _vh)
{  return VertexEdgeIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeCWIter PolyConnectivity::ve_cwiter(ArrayKernel::VertexHandle _vh)
{  return VertexEdgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeCCWIter PolyConnectivity::ve_ccwiter(ArrayKernel::VertexHandle _vh)
{  return VertexEdgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexFaceIter PolyConnectivity::vf_iter(ArrayKernel::VertexHandle _vh)
{  return VertexFaceIter(*this, _vh); }

inline PolyConnectivity::VertexFaceCWIter PolyConnectivity::vf_cwiter(ArrayKernel::VertexHandle _vh)
{  return VertexFaceCWIter(*this, _vh); }

inline PolyConnectivity::VertexFaceCCWIter PolyConnectivity::vf_ccwiter(ArrayKernel::VertexHandle _vh)
{  return VertexFaceCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexVertexIter PolyConnectivity::cvv_iter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexVertexIter(*this, _vh); }

inline PolyConnectivity::ConstVertexVertexCWIter PolyConnectivity::cvv_cwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexVertexCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexVertexCCWIter PolyConnectivity::cvv_ccwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexVertexCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeIter PolyConnectivity::cvih_iter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexIHalfedgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeCWIter PolyConnectivity::cvih_cwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexIHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeCCWIter PolyConnectivity::cvih_ccwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexIHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeIter PolyConnectivity::cvoh_iter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexOHalfedgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeCWIter PolyConnectivity::cvoh_cwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexOHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeCCWIter PolyConnectivity::cvoh_ccwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexOHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeIter PolyConnectivity::cve_iter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexEdgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeCWIter PolyConnectivity::cve_cwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexEdgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeCCWIter PolyConnectivity::cve_ccwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexEdgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceIter PolyConnectivity::cvf_iter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexFaceIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceCWIter PolyConnectivity::cvf_cwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexFaceCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceCCWIter PolyConnectivity::cvf_ccwiter(ArrayKernel::VertexHandle _vh) const
{ return ConstVertexFaceCCWIter(*this, _vh); }

inline PolyConnectivity::FaceVertexIter PolyConnectivity::fv_iter(ArrayKernel::FaceHandle _fh)
{ return FaceVertexIter(*this, _fh); }

inline PolyConnectivity::FaceVertexCWIter PolyConnectivity::fv_cwiter(ArrayKernel::FaceHandle _fh)
{ return FaceVertexCWIter(*this, _fh); }

inline PolyConnectivity::FaceVertexCCWIter PolyConnectivity::fv_ccwiter(ArrayKernel::FaceHandle _fh)
{ return FaceVertexCCWIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeIter PolyConnectivity::fh_iter(ArrayKernel::FaceHandle _fh)
{ return FaceHalfedgeIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeCWIter PolyConnectivity::fh_cwiter(ArrayKernel::FaceHandle _fh)
{ return FaceHalfedgeCWIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeCCWIter PolyConnectivity::fh_ccwiter(ArrayKernel::FaceHandle _fh)
{ return FaceHalfedgeCCWIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeIter PolyConnectivity::fe_iter(ArrayKernel::FaceHandle _fh)
{ return FaceEdgeIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeCWIter PolyConnectivity::fe_cwiter(ArrayKernel::FaceHandle _fh)
{ return FaceEdgeCWIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeCCWIter PolyConnectivity::fe_ccwiter(ArrayKernel::FaceHandle _fh)
{ return FaceEdgeCCWIter(*this, _fh); }

inline PolyConnectivity::FaceFaceIter PolyConnectivity::ff_iter(ArrayKernel::FaceHandle _fh)
{ return FaceFaceIter(*this, _fh); }

inline PolyConnectivity::FaceFaceCWIter PolyConnectivity::ff_cwiter(ArrayKernel::FaceHandle _fh)
{ return FaceFaceCWIter(*this, _fh); }

inline PolyConnectivity::FaceFaceCCWIter PolyConnectivity::ff_ccwiter(ArrayKernel::FaceHandle _fh)
{ return FaceFaceCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceVertexIter PolyConnectivity::cfv_iter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceVertexIter(*this, _fh); }

inline PolyConnectivity::ConstFaceVertexCWIter PolyConnectivity::cfv_cwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceVertexCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceVertexCCWIter PolyConnectivity::cfv_ccwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceVertexCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeIter PolyConnectivity::cfh_iter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceHalfedgeIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeCWIter PolyConnectivity::cfh_cwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceHalfedgeCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeCCWIter PolyConnectivity::cfh_ccwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceHalfedgeCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeIter PolyConnectivity::cfe_iter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceEdgeIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeCWIter PolyConnectivity::cfe_cwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceEdgeCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeCCWIter PolyConnectivity::cfe_ccwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceEdgeCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceIter PolyConnectivity::cff_iter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceFaceIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceCWIter PolyConnectivity::cff_cwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceFaceCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceCCWIter PolyConnectivity::cff_ccwiter(ArrayKernel::FaceHandle _fh) const
{ return ConstFaceFaceCCWIter(*this, _fh); }

inline PolyConnectivity::EdgeVertexIter PolyConnectivity::ev_iter(ArrayKernel::EdgeHandle _eh)
{ return EdgeVertexIter(*this, _eh); }

inline PolyConnectivity::EdgeHalfedgeIter PolyConnectivity::eh_iter(ArrayKernel::EdgeHandle _eh)
{ return EdgeHalfedgeIter(*this, _eh); }

inline PolyConnectivity::EdgeFaceIter PolyConnectivity::ef_iter(ArrayKernel::EdgeHandle _eh)
{ return EdgeFaceIter(*this, _eh); }

inline PolyConnectivity::ConstEdgeVertexIter PolyConnectivity::cev_iter(ArrayKernel::EdgeHandle _eh) const
{ return ConstEdgeVertexIter(*this, _eh); }

inline PolyConnectivity::ConstEdgeHalfedgeIter PolyConnectivity::ceh_iter(ArrayKernel::EdgeHandle _eh) const
{ return ConstEdgeHalfedgeIter(*this, _eh); }

inline PolyConnectivity::ConstEdgeFaceIter PolyConnectivity::cef_iter(ArrayKernel::EdgeHandle _eh) const
{ return ConstEdgeFaceIter(*this, _eh); }


inline PolyConnectivity::VertexVertexIter PolyConnectivity::vv_begin(VertexHandle _vh)
{ return VertexVertexIter(*this, _vh); }

inline PolyConnectivity::VertexVertexCWIter PolyConnectivity::vv_cwbegin(VertexHandle _vh)
{ return VertexVertexCWIter(*this, _vh); }

inline PolyConnectivity::VertexVertexCCWIter PolyConnectivity::vv_ccwbegin(VertexHandle _vh)
{ return VertexVertexCCWIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeIter PolyConnectivity::vih_begin(VertexHandle _vh)
{ return VertexIHalfedgeIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeCWIter PolyConnectivity::vih_cwbegin(VertexHandle _vh)
{ return VertexIHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexIHalfedgeCCWIter PolyConnectivity::vih_ccwbegin(VertexHandle _vh)
{ return VertexIHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeIter PolyConnectivity::voh_begin(VertexHandle _vh)
{ return VertexOHalfedgeIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeCWIter PolyConnectivity::voh_cwbegin(VertexHandle _vh)
{ return VertexOHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexOHalfedgeCCWIter PolyConnectivity::voh_ccwbegin(VertexHandle _vh)
{ return VertexOHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeIter PolyConnectivity::ve_begin(VertexHandle _vh)
{ return VertexEdgeIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeCWIter PolyConnectivity::ve_cwbegin(VertexHandle _vh)
{ return VertexEdgeCWIter(*this, _vh); }

inline PolyConnectivity::VertexEdgeCCWIter PolyConnectivity::ve_ccwbegin(VertexHandle _vh)
{ return VertexEdgeCCWIter(*this, _vh); }

inline PolyConnectivity::VertexFaceIter PolyConnectivity::vf_begin(VertexHandle _vh)
{ return VertexFaceIter(*this, _vh); }

inline PolyConnectivity::VertexFaceCWIter PolyConnectivity::vf_cwbegin(VertexHandle _vh)
{ return VertexFaceCWIter(*this, _vh); }

inline PolyConnectivity::VertexFaceCCWIter PolyConnectivity::vf_ccwbegin(VertexHandle _vh)
{ return VertexFaceCCWIter(*this, _vh); }


inline PolyConnectivity::ConstVertexVertexIter PolyConnectivity::cvv_begin(VertexHandle _vh) const
{ return ConstVertexVertexIter(*this, _vh); }

inline PolyConnectivity::ConstVertexVertexCWIter PolyConnectivity::cvv_cwbegin(VertexHandle _vh) const
{ return ConstVertexVertexCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexVertexCCWIter PolyConnectivity::cvv_ccwbegin(VertexHandle _vh) const
{ return ConstVertexVertexCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeIter PolyConnectivity::cvih_begin(VertexHandle _vh) const
{ return ConstVertexIHalfedgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeCWIter PolyConnectivity::cvih_cwbegin(VertexHandle _vh) const
{ return ConstVertexIHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexIHalfedgeCCWIter PolyConnectivity::cvih_ccwbegin(VertexHandle _vh) const
{ return ConstVertexIHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeIter PolyConnectivity::cvoh_begin(VertexHandle _vh) const
{ return ConstVertexOHalfedgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeCWIter PolyConnectivity::cvoh_cwbegin(VertexHandle _vh) const
{ return ConstVertexOHalfedgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexOHalfedgeCCWIter PolyConnectivity::cvoh_ccwbegin(VertexHandle _vh) const
{ return ConstVertexOHalfedgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeIter PolyConnectivity::cve_begin(VertexHandle _vh) const
{ return ConstVertexEdgeIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeCWIter PolyConnectivity::cve_cwbegin(VertexHandle _vh) const
{ return ConstVertexEdgeCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexEdgeCCWIter PolyConnectivity::cve_ccwbegin(VertexHandle _vh) const
{ return ConstVertexEdgeCCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceIter PolyConnectivity::cvf_begin(VertexHandle _vh) const
{ return ConstVertexFaceIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceCWIter PolyConnectivity::cvf_cwbegin(VertexHandle _vh) const
{ return ConstVertexFaceCWIter(*this, _vh); }

inline PolyConnectivity::ConstVertexFaceCCWIter PolyConnectivity::cvf_ccwbegin(VertexHandle _vh) const
{ return ConstVertexFaceCCWIter(*this, _vh); }


inline PolyConnectivity::FaceVertexIter PolyConnectivity::fv_begin(FaceHandle _fh)
{ return FaceVertexIter(*this, _fh); }

inline PolyConnectivity::FaceVertexCWIter PolyConnectivity::fv_cwbegin(FaceHandle _fh)
{ return FaceVertexCWIter(*this, _fh); }

inline PolyConnectivity::FaceVertexCCWIter PolyConnectivity::fv_ccwbegin(FaceHandle _fh)
{ return FaceVertexCCWIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeIter PolyConnectivity::fh_begin(FaceHandle _fh)
{ return FaceHalfedgeIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeCWIter PolyConnectivity::fh_cwbegin(FaceHandle _fh)
{ return FaceHalfedgeCWIter(*this, _fh); }

inline PolyConnectivity::FaceHalfedgeCCWIter PolyConnectivity::fh_ccwbegin(FaceHandle _fh)
{ return FaceHalfedgeCCWIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeIter PolyConnectivity::fe_begin(FaceHandle _fh)
{ return FaceEdgeIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeCWIter PolyConnectivity::fe_cwbegin(FaceHandle _fh)
{ return FaceEdgeCWIter(*this, _fh); }

inline PolyConnectivity::FaceEdgeCCWIter PolyConnectivity::fe_ccwbegin(FaceHandle _fh)
{ return FaceEdgeCCWIter(*this, _fh); }

inline PolyConnectivity::FaceFaceIter PolyConnectivity::ff_begin(FaceHandle _fh)
{ return FaceFaceIter(*this, _fh); }

inline PolyConnectivity::FaceFaceCWIter PolyConnectivity::ff_cwbegin(FaceHandle _fh)
{ return FaceFaceCWIter(*this, _fh); }

inline PolyConnectivity::FaceFaceCCWIter PolyConnectivity::ff_ccwbegin(FaceHandle _fh)
{ return FaceFaceCCWIter(*this, _fh); }

inline PolyConnectivity::HalfedgeLoopIter PolyConnectivity::hl_begin(HalfedgeHandle _heh)
{ return HalfedgeLoopIter(*this, _heh); }

inline PolyConnectivity::HalfedgeLoopCWIter PolyConnectivity::hl_cwbegin(HalfedgeHandle _heh)
{ return HalfedgeLoopCWIter(*this, _heh); }

inline PolyConnectivity::HalfedgeLoopCCWIter PolyConnectivity::hl_ccwbegin(HalfedgeHandle _heh)
{ return HalfedgeLoopCCWIter(*this, _heh); }


inline PolyConnectivity::ConstFaceVertexIter PolyConnectivity::cfv_begin(FaceHandle _fh) const
{ return ConstFaceVertexIter(*this, _fh); }

inline PolyConnectivity::ConstFaceVertexCWIter PolyConnectivity::cfv_cwbegin(FaceHandle _fh) const
{ return ConstFaceVertexCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceVertexCCWIter PolyConnectivity::cfv_ccwbegin(FaceHandle _fh) const
{ return ConstFaceVertexCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeIter PolyConnectivity::cfh_begin(FaceHandle _fh) const
{ return ConstFaceHalfedgeIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeCWIter PolyConnectivity::cfh_cwbegin(FaceHandle _fh) const
{ return ConstFaceHalfedgeCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceHalfedgeCCWIter PolyConnectivity::cfh_ccwbegin(FaceHandle _fh) const
{ return ConstFaceHalfedgeCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeIter PolyConnectivity::cfe_begin(FaceHandle _fh) const
{ return ConstFaceEdgeIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeCWIter PolyConnectivity::cfe_cwbegin(FaceHandle _fh) const
{ return ConstFaceEdgeCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceEdgeCCWIter PolyConnectivity::cfe_ccwbegin(FaceHandle _fh) const
{ return ConstFaceEdgeCCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceIter PolyConnectivity::cff_begin(FaceHandle _fh) const
{ return ConstFaceFaceIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceCWIter PolyConnectivity::cff_cwbegin(FaceHandle _fh) const
{ return ConstFaceFaceCWIter(*this, _fh); }

inline PolyConnectivity::ConstFaceFaceCCWIter PolyConnectivity::cff_ccwbegin(FaceHandle _fh) const
{ return ConstFaceFaceCCWIter(*this, _fh); }

inline PolyConnectivity::ConstHalfedgeLoopIter PolyConnectivity::chl_begin(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopIter(*this, _heh); }

inline PolyConnectivity::ConstHalfedgeLoopCWIter PolyConnectivity::chl_cwbegin(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopCWIter(*this, _heh); }

inline PolyConnectivity::ConstHalfedgeLoopCCWIter PolyConnectivity::chl_ccwbegin(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopCCWIter(*this, _heh); }


inline PolyConnectivity::EdgeVertexIter PolyConnectivity::ev_begin(EdgeHandle _eh)
{ return EdgeVertexIter(*this, _eh); }

inline PolyConnectivity::EdgeHalfedgeIter PolyConnectivity::eh_begin(EdgeHandle _eh)
{ return EdgeHalfedgeIter(*this, _eh); }

inline PolyConnectivity::EdgeFaceIter PolyConnectivity::ef_begin(EdgeHandle _eh)
{ return EdgeFaceIter(*this, _eh); }


inline PolyConnectivity::ConstEdgeVertexIter PolyConnectivity::cev_begin(EdgeHandle _eh) const
{ return ConstEdgeVertexIter(*this, _eh); }

inline PolyConnectivity::ConstEdgeHalfedgeIter PolyConnectivity::ceh_begin(EdgeHandle _eh) const
{ return ConstEdgeHalfedgeIter(*this, _eh); }

inline PolyConnectivity::ConstEdgeFaceIter PolyConnectivity::cef_begin(EdgeHandle _eh) const
{ return ConstEdgeFaceIter(*this, _eh); }


// 'end' circulators

inline PolyConnectivity::VertexVertexIter PolyConnectivity::vv_end(VertexHandle _vh)
{ return VertexVertexIter(*this, _vh, true); }

inline PolyConnectivity::VertexVertexCWIter PolyConnectivity::vv_cwend(VertexHandle _vh)
{ return VertexVertexCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexVertexCCWIter PolyConnectivity::vv_ccwend(VertexHandle _vh)
{ return VertexVertexCCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexIHalfedgeIter PolyConnectivity::vih_end(VertexHandle _vh)
{ return VertexIHalfedgeIter(*this, _vh, true); }

inline PolyConnectivity::VertexIHalfedgeCWIter PolyConnectivity::vih_cwend(VertexHandle _vh)
{ return VertexIHalfedgeCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexIHalfedgeCCWIter PolyConnectivity::vih_ccwend(VertexHandle _vh)
{ return VertexIHalfedgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexOHalfedgeIter PolyConnectivity::voh_end(VertexHandle _vh)
{ return VertexOHalfedgeIter(*this, _vh, true); }

inline PolyConnectivity::VertexOHalfedgeCWIter PolyConnectivity::voh_cwend(VertexHandle _vh)
{ return VertexOHalfedgeCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexOHalfedgeCCWIter PolyConnectivity::voh_ccwend(VertexHandle _vh)
{ return VertexOHalfedgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexEdgeIter PolyConnectivity::ve_end(VertexHandle _vh)
{ return VertexEdgeIter(*this, _vh, true); }

inline PolyConnectivity::VertexEdgeCWIter PolyConnectivity::ve_cwend(VertexHandle _vh)
{ return VertexEdgeCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexEdgeCCWIter PolyConnectivity::ve_ccwend(VertexHandle _vh)
{ return VertexEdgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexFaceIter PolyConnectivity::vf_end(VertexHandle _vh)
{ return VertexFaceIter(*this, _vh, true); }

inline PolyConnectivity::VertexFaceCWIter PolyConnectivity::vf_cwend(VertexHandle _vh)
{ return VertexFaceCWIter(*this, _vh, true); }

inline PolyConnectivity::VertexFaceCCWIter PolyConnectivity::vf_ccwend(VertexHandle _vh)
{ return VertexFaceCCWIter(*this, _vh, true); }


inline PolyConnectivity::ConstVertexVertexIter PolyConnectivity::cvv_end(VertexHandle _vh) const
{ return ConstVertexVertexIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexVertexCWIter PolyConnectivity::cvv_cwend(VertexHandle _vh) const
{ return ConstVertexVertexCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexVertexCCWIter PolyConnectivity::cvv_ccwend(VertexHandle _vh) const
{ return ConstVertexVertexCCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexIHalfedgeIter PolyConnectivity::cvih_end(VertexHandle _vh) const
{ return ConstVertexIHalfedgeIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexIHalfedgeCWIter PolyConnectivity::cvih_cwend(VertexHandle _vh) const
{ return ConstVertexIHalfedgeCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexIHalfedgeCCWIter PolyConnectivity::cvih_ccwend(VertexHandle _vh) const
{ return ConstVertexIHalfedgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexOHalfedgeIter PolyConnectivity::cvoh_end(VertexHandle _vh) const
{ return ConstVertexOHalfedgeIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexOHalfedgeCWIter PolyConnectivity::cvoh_cwend(VertexHandle _vh) const
{ return ConstVertexOHalfedgeCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexOHalfedgeCCWIter PolyConnectivity::cvoh_ccwend(VertexHandle _vh) const
{ return ConstVertexOHalfedgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexEdgeIter PolyConnectivity::cve_end(VertexHandle _vh) const
{ return ConstVertexEdgeIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexEdgeCWIter PolyConnectivity::cve_cwend(VertexHandle _vh) const
{ return ConstVertexEdgeCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexEdgeCCWIter PolyConnectivity::cve_ccwend(VertexHandle _vh) const
{ return ConstVertexEdgeCCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexFaceIter PolyConnectivity::cvf_end(VertexHandle _vh) const
{ return ConstVertexFaceIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexFaceCWIter PolyConnectivity::cvf_cwend(VertexHandle _vh) const
{ return ConstVertexFaceCWIter(*this, _vh, true); }

inline PolyConnectivity::ConstVertexFaceCCWIter PolyConnectivity::cvf_ccwend(VertexHandle _vh) const
{ return ConstVertexFaceCCWIter(*this, _vh, true); }


inline PolyConnectivity::FaceVertexIter PolyConnectivity::fv_end(FaceHandle _fh)
{ return FaceVertexIter(*this, _fh, true); }

inline PolyConnectivity::FaceVertexCWIter PolyConnectivity::fv_cwend(FaceHandle _fh)
{ return FaceVertexCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceVertexCCWIter PolyConnectivity::fv_ccwend(FaceHandle _fh)
{ return FaceVertexCCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceHalfedgeIter PolyConnectivity::fh_end(FaceHandle _fh)
{ return FaceHalfedgeIter(*this, _fh, true); }

inline PolyConnectivity::FaceHalfedgeCWIter PolyConnectivity::fh_cwend(FaceHandle _fh)
{ return FaceHalfedgeCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceHalfedgeCCWIter PolyConnectivity::fh_ccwend(FaceHandle _fh)
{ return FaceHalfedgeCCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceEdgeIter PolyConnectivity::fe_end(FaceHandle _fh)
{ return FaceEdgeIter(*this, _fh, true); }

inline PolyConnectivity::FaceEdgeCWIter PolyConnectivity::fe_cwend(FaceHandle _fh)
{ return FaceEdgeCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceEdgeCCWIter PolyConnectivity::fe_ccwend(FaceHandle _fh)
{ return FaceEdgeCCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceFaceIter PolyConnectivity::ff_end(FaceHandle _fh)
{ return FaceFaceIter(*this, _fh, true); }

inline PolyConnectivity::FaceFaceCWIter PolyConnectivity::ff_cwend(FaceHandle _fh)
{ return FaceFaceCWIter(*this, _fh, true); }

inline PolyConnectivity::FaceFaceCCWIter PolyConnectivity::ff_ccwend(FaceHandle _fh)
{ return FaceFaceCCWIter(*this, _fh, true); }

inline PolyConnectivity::HalfedgeLoopIter PolyConnectivity::hl_end(HalfedgeHandle _heh)
{ return HalfedgeLoopIter(*this, _heh, true); }

inline PolyConnectivity::HalfedgeLoopCWIter PolyConnectivity::hl_cwend(HalfedgeHandle _heh)
{ return HalfedgeLoopCWIter(*this, _heh, true); }

inline PolyConnectivity::HalfedgeLoopCCWIter PolyConnectivity::hl_ccwend(HalfedgeHandle _heh)
{ return HalfedgeLoopCCWIter(*this, _heh, true); }


inline PolyConnectivity::ConstFaceVertexIter PolyConnectivity::cfv_end(FaceHandle _fh) const
{ return ConstFaceVertexIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceVertexCWIter PolyConnectivity::cfv_cwend(FaceHandle _fh) const
{ return ConstFaceVertexCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceVertexCCWIter PolyConnectivity::cfv_ccwend(FaceHandle _fh) const
{ return ConstFaceVertexCCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceHalfedgeIter PolyConnectivity::cfh_end(FaceHandle _fh) const
{ return ConstFaceHalfedgeIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceHalfedgeCWIter PolyConnectivity::cfh_cwend(FaceHandle _fh) const
{ return ConstFaceHalfedgeCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceHalfedgeCCWIter PolyConnectivity::cfh_ccwend(FaceHandle _fh) const
{ return ConstFaceHalfedgeCCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceEdgeIter PolyConnectivity::cfe_end(FaceHandle _fh) const
{ return ConstFaceEdgeIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceEdgeCWIter PolyConnectivity::cfe_cwend(FaceHandle _fh) const
{ return ConstFaceEdgeCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceEdgeCCWIter PolyConnectivity::cfe_ccwend(FaceHandle _fh) const
{ return ConstFaceEdgeCCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceFaceIter PolyConnectivity::cff_end(FaceHandle _fh) const
{ return ConstFaceFaceIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceFaceCWIter PolyConnectivity::cff_cwend(FaceHandle _fh) const
{ return ConstFaceFaceCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstFaceFaceCCWIter PolyConnectivity::cff_ccwend(FaceHandle _fh) const
{ return ConstFaceFaceCCWIter(*this, _fh, true); }

inline PolyConnectivity::ConstHalfedgeLoopIter PolyConnectivity::chl_end(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopIter(*this, _heh, true); }

inline PolyConnectivity::ConstHalfedgeLoopCWIter PolyConnectivity::chl_cwend(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopCWIter(*this, _heh, true); }

inline PolyConnectivity::ConstHalfedgeLoopCCWIter PolyConnectivity::chl_ccwend(HalfedgeHandle _heh) const
{ return ConstHalfedgeLoopCCWIter(*this, _heh, true); }


inline PolyConnectivity::EdgeVertexIter PolyConnectivity::ev_end(EdgeHandle _eh)
{ return EdgeVertexIter(*this, _eh, true); }

inline PolyConnectivity::EdgeHalfedgeIter PolyConnectivity::eh_end(EdgeHandle _eh)
{ return EdgeHalfedgeIter(*this, _eh, true); }

inline PolyConnectivity::EdgeFaceIter PolyConnectivity::ef_end(EdgeHandle _eh)
{ return EdgeFaceIter(*this, _eh, true); }


inline PolyConnectivity::ConstEdgeVertexIter PolyConnectivity::cev_end(EdgeHandle _eh) const
{ return ConstEdgeVertexIter(*this, _eh, true); }

inline PolyConnectivity::ConstEdgeHalfedgeIter PolyConnectivity::ceh_end(EdgeHandle _eh) const
{ return ConstEdgeHalfedgeIter(*this, _eh, true); }

inline PolyConnectivity::ConstEdgeFaceIter PolyConnectivity::cef_end(EdgeHandle _eh) const
{ return ConstEdgeFaceIter(*this, _eh, true); }


inline PolyConnectivity::ConstVertexFaceRange         SmartVertexHandle::faces()                                     const { assert(mesh() != nullptr); return mesh()->vf_range    (*this); }
inline PolyConnectivity::ConstVertexFaceCWRange       SmartVertexHandle::faces_cw()                                  const { assert(mesh() != nullptr); return mesh()->vf_cw_range (*this); }
inline PolyConnectivity::ConstVertexFaceCCWRange      SmartVertexHandle::faces_ccw()                                 const { assert(mesh() != nullptr); return mesh()->vf_ccw_range(*this); }

inline PolyConnectivity::ConstVertexEdgeRange         SmartVertexHandle::edges()                                     const { assert(mesh() != nullptr); return mesh()->ve_range    (*this); }
inline PolyConnectivity::ConstVertexEdgeCWRange       SmartVertexHandle::edges_cw()                                  const { assert(mesh() != nullptr); return mesh()->ve_cw_range (*this); }
inline PolyConnectivity::ConstVertexEdgeCCWRange      SmartVertexHandle::edges_ccw()                                 const { assert(mesh() != nullptr); return mesh()->ve_ccw_range(*this); }

inline PolyConnectivity::ConstVertexVertexRange       SmartVertexHandle::vertices()                                  const { assert(mesh() != nullptr); return mesh()->vv_range     (*this); }
inline PolyConnectivity::ConstVertexVertexCWRange     SmartVertexHandle::vertices_cw()                               const { assert(mesh() != nullptr); return mesh()->vv_cw_range  (*this); }
inline PolyConnectivity::ConstVertexVertexCCWRange    SmartVertexHandle::vertices_ccw()                              const { assert(mesh() != nullptr); return mesh()->vv_ccw_range (*this); }

inline PolyConnectivity::ConstVertexIHalfedgeRange    SmartVertexHandle::incoming_halfedges()                        const { assert(mesh() != nullptr); return mesh()->vih_range    (*this); }
inline PolyConnectivity::ConstVertexIHalfedgeCWRange  SmartVertexHandle::incoming_halfedges_cw()                     const { assert(mesh() != nullptr); return mesh()->vih_cw_range (*this); }
inline PolyConnectivity::ConstVertexIHalfedgeCCWRange SmartVertexHandle::incoming_halfedges_ccw()                    const { assert(mesh() != nullptr); return mesh()->vih_ccw_range(*this); }

inline PolyConnectivity::ConstVertexIHalfedgeRange    SmartVertexHandle::incoming_halfedges    (HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->vih_range    (_heh);  }
inline PolyConnectivity::ConstVertexIHalfedgeCWRange  SmartVertexHandle::incoming_halfedges_cw (HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->vih_cw_range (_heh);  }
inline PolyConnectivity::ConstVertexIHalfedgeCCWRange SmartVertexHandle::incoming_halfedges_ccw(HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->vih_ccw_range(_heh);  }

inline PolyConnectivity::ConstVertexOHalfedgeRange    SmartVertexHandle::outgoing_halfedges()                        const { assert(mesh() != nullptr); return mesh()->voh_range    (*this); }
inline PolyConnectivity::ConstVertexOHalfedgeCWRange  SmartVertexHandle::outgoing_halfedges_cw()                     const { assert(mesh() != nullptr); return mesh()->voh_cw_range (*this); }
inline PolyConnectivity::ConstVertexOHalfedgeCCWRange SmartVertexHandle::outgoing_halfedges_ccw()                    const { assert(mesh() != nullptr); return mesh()->voh_ccw_range(*this); }

inline PolyConnectivity::ConstVertexOHalfedgeRange    SmartVertexHandle::outgoing_halfedges    (HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->voh_range    (_heh);  }
inline PolyConnectivity::ConstVertexOHalfedgeCWRange  SmartVertexHandle::outgoing_halfedges_cw (HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->voh_cw_range (_heh);  }
inline PolyConnectivity::ConstVertexOHalfedgeCCWRange SmartVertexHandle::outgoing_halfedges_ccw(HalfedgeHandle _heh) const { assert(mesh() != nullptr); return mesh()->voh_ccw_range(_heh);  }


inline PolyConnectivity::ConstHalfedgeLoopRange       SmartHalfedgeHandle::loop()                                    const { assert(mesh() != nullptr); return mesh()->hl_range     (*this); }
inline PolyConnectivity::ConstHalfedgeLoopCWRange     SmartHalfedgeHandle::loop_cw()                                 const { assert(mesh() != nullptr); return mesh()->hl_cw_range  (*this); }
inline PolyConnectivity::ConstHalfedgeLoopCCWRange    SmartHalfedgeHandle::loop_ccw()                                const { assert(mesh() != nullptr); return mesh()->hl_ccw_range (*this); }


inline PolyConnectivity::ConstFaceVertexRange         SmartFaceHandle::vertices()                                    const { assert(mesh() != nullptr); return mesh()->fv_range    (*this); }
inline PolyConnectivity::ConstFaceVertexCWRange       SmartFaceHandle::vertices_cw()                                 const { assert(mesh() != nullptr); return mesh()->fv_cw_range (*this); }
inline PolyConnectivity::ConstFaceVertexCCWRange      SmartFaceHandle::vertices_ccw()                                const { assert(mesh() != nullptr); return mesh()->fv_ccw_range(*this); }

inline PolyConnectivity::ConstFaceHalfedgeRange       SmartFaceHandle::halfedges()                                   const { assert(mesh() != nullptr); return mesh()->fh_range    (*this); }
inline PolyConnectivity::ConstFaceHalfedgeCWRange     SmartFaceHandle::halfedges_cw()                                const { assert(mesh() != nullptr); return mesh()->fh_cw_range (*this); }
inline PolyConnectivity::ConstFaceHalfedgeCCWRange    SmartFaceHandle::halfedges_ccw()                               const { assert(mesh() != nullptr); return mesh()->fh_ccw_range(*this); }

inline PolyConnectivity::ConstFaceEdgeRange           SmartFaceHandle::edges()                                       const { assert(mesh() != nullptr); return mesh()->fe_range    (*this); }
inline PolyConnectivity::ConstFaceEdgeCWRange         SmartFaceHandle::edges_cw()                                    const { assert(mesh() != nullptr); return mesh()->fe_cw_range (*this); }
inline PolyConnectivity::ConstFaceEdgeCCWRange        SmartFaceHandle::edges_ccw()                                   const { assert(mesh() != nullptr); return mesh()->fe_ccw_range(*this); }

inline PolyConnectivity::ConstFaceFaceRange           SmartFaceHandle::faces()                                       const { assert(mesh() != nullptr); return mesh()->ff_range    (*this); }
inline PolyConnectivity::ConstFaceFaceCWRange         SmartFaceHandle::faces_cw()                                    const { assert(mesh() != nullptr); return mesh()->ff_cw_range (*this); }
inline PolyConnectivity::ConstFaceFaceCCWRange        SmartFaceHandle::faces_ccw()                                   const { assert(mesh() != nullptr); return mesh()->ff_ccw_range(*this); }


inline PolyConnectivity::ConstEdgeVertexRange         SmartEdgeHandle::vertices()                                    const { assert(mesh() != nullptr); return mesh()->ev_range    (*this); }

inline PolyConnectivity::ConstEdgeHalfedgeRange       SmartEdgeHandle::halfedges()                                   const { assert(mesh() != nullptr); return mesh()->eh_range    (*this); }

inline PolyConnectivity::ConstEdgeHalfedgeRange       SmartEdgeHandle::halfedges(HalfedgeHandle _heh)                const { assert(mesh() != nullptr); return mesh()->eh_range    (_heh);  }

inline PolyConnectivity::ConstEdgeFaceRange           SmartEdgeHandle::faces()                                       const { assert(mesh() != nullptr); return mesh()->ef_range    (*this); }

}//namespace OpenMesh
