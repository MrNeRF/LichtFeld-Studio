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




#define OPENMESH_MESHCHECKER_C


//== INCLUDES =================================================================


#include <OpenMesh/Tools/Utils/MeshCheckerT.hh>


//== NAMESPACES ============================================================== 


namespace OpenMesh {
namespace Utils {

//== IMPLEMENTATION ========================================================== 


template <class Mesh>
bool 
MeshCheckerT<Mesh>::
check(unsigned int _targets, std::ostream& _os)
{
  bool  ok(true);



  //--- vertex checks ---

  if (_targets & CHECK_VERTICES)
  {
    unsigned int                   count;
    const unsigned int             max_valence(10000);


    for (const auto vh: mesh_.vertices())
    {
        /* The outgoing halfedge of a boundary vertex has to be a boundary halfedge */
        auto heh = vh.halfedge();
        if (heh.is_valid() && !mesh_.is_boundary(heh))
        {
          for (typename Mesh::ConstVertexOHalfedgeIter vh_it(mesh_, vh);
              vh_it.is_valid(); ++vh_it)
          {
            if (mesh_.is_boundary(*vh_it))
            {
              _os << "MeshChecker: vertex " << vh
                  << ": outgoing halfedge not on boundary error\n";
              ok = false;
            }
          }
        }
        if (heh.is_valid()) {
          if (heh.idx() < -1 || heh.idx() >= (int)mesh_.n_halfedges()) {
              _os << "MeshChecker: vertex " << vh
                  << " has out-of-bounds outgoing HE: " << heh;
              ok = false;
            }
            if (is_deleted(heh.edge())) {
              _os << "MeshChecker: vertex " << vh
                  << " has deleted outgoing HE: " << heh;
              ok = false;
            }
        }


      
        // outgoing halfedge has to refer back to vertex
        if (mesh_.halfedge_handle(vh).is_valid() &&
            mesh_.from_vertex_handle(mesh_.halfedge_handle(vh)) != vh)
        {
          _os << "MeshChecker: vertex " << vh
              << ": outgoing halfedge does not reference vertex\n";
          ok = false;
        }


        // check whether circulators are still in order
        auto vv_it = mesh_.cvv_cwiter(vh);
        for (count=0; vv_it.is_valid() && (count < max_valence); ++vv_it, ++count) {};
        if (count == max_valence)
        {
          _os << "MeshChecker: vertex " << vh
              << ": ++circulator problem, one ring corrupt\n";
          ok = false;
        }
        vv_it = mesh_.cvv_cwiter(vh);
        for (count=0; vv_it.is_valid() && (count < max_valence); --vv_it, ++count) {};
        if (count == max_valence)
        {
          _os << "MeshChecker: vertex " << vh
              << ": --circulator problem, one ring corrupt\n";
          ok = false;
        }
    }
  }



  //--- halfedge checks ---

  if (_targets & CHECK_EDGES)
  {
    typename Mesh::HalfedgeHandle     hstart, hhh;
    size_t                            n_halfedges = 2*mesh_.n_edges();

    for (const auto hh: mesh_.halfedges())
    {
      if (!hh.to().halfedge().is_valid()) {
            _os << "MeshChecker: vertex " << hh.from()
                << " has no outgoing halfedge, but it is not isolated.\n";
            ok = false;
      }
      // degenerated halfedge ?
      if (mesh_.from_vertex_handle(hh) == mesh_.to_vertex_handle(hh))
      {
        _os << "MeshChecker: halfedge " << hh
            << ": to-vertex == from-vertex\n";
        ok = false;
      }


      // next <-> prev check
      if (mesh_.next_halfedge_handle(mesh_.prev_halfedge_handle(hh)) != hh)
      {
        _os << "MeshChecker: halfedge " << hh
            << ": prev->next != this\n";
        ok = false;
      }

      // heh.to == heh.next.from?
      if (mesh_.to_vertex_handle(hh) != mesh_.from_vertex_handle(
                  mesh_.next_halfedge_handle(hh)))
      {
        _os << "MeshChecker: halfedge " << hh
            << ".to != he.next.from\n";
        ok = false;
      }
      // heh.from == heh.prev.to?
      if (mesh_.from_vertex_handle(hh) != mesh_.to_vertex_handle(
                  mesh_.prev_halfedge_handle(hh)))
      {
        _os << "MeshChecker: halfedge " << hh
            << ".from != he.prev.to\n";
        ok = false;
      }


      // halfedges should form a cycle
      size_t count=0; hstart=hhh=hh;
      do
      {
        hhh = mesh_.next_halfedge_handle(hhh);
        ++count;
      } while (hhh != hstart && count < n_halfedges);

      if (count == n_halfedges)
      {
        _os << "MeshChecker: halfedges starting from " << hh
            << " do not form a cycle\n";
        ok = false;
      }
    }
  }



  //--- face checks ---

  if (_targets & CHECK_FACES)
  {
    typename Mesh::ConstFaceHalfedgeIter  fh_it;

    for(const auto fh: mesh_.faces()) {
        for(const auto heh: fh.halfedges()) {
            if (heh.face() != fh) {
                _os << "MeshChecker: face " << fh
                    << ": its halfedge " << heh << " references a different face: "
                    << heh.face()
                    << ".\n";
                ok = false;
            }
        }
    }
  }



  return ok;
}


//=============================================================================
} // naespace Utils
} // namespace OpenMesh
//=============================================================================
