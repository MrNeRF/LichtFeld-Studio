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


/** \file MixedDecimaterT.hh
 */

//=============================================================================
//
//  CLASS MixedDecimaterT - IMPLEMENTATION
//
//=============================================================================

#ifndef OPENMESH_MIXED_DECIMATER_DECIMATERT_HH
#define OPENMESH_MIXED_DECIMATER_DECIMATERT_HH


//== INCLUDES =================================================================

#include <memory>
#include <OpenMesh/Tools/Decimater/McDecimaterT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>



//== NAMESPACE ================================================================

namespace OpenMesh  {
namespace Decimater {


//== CLASS DEFINITION =========================================================


/** Mixed decimater framework
    \see BaseModT, \ref decimater_docu
*/
template < typename MeshT >
class MixedDecimaterT : public McDecimaterT<MeshT>, public DecimaterT<MeshT>
{
public: //-------------------------------------------------------- public types

  typedef McDecimaterT< MeshT >         Self;
  typedef MeshT                         Mesh;
  typedef CollapseInfoT<MeshT>          CollapseInfo;
  typedef ModBaseT<MeshT>               Module;
  typedef std::vector< Module* >        ModuleList;
  typedef typename ModuleList::iterator ModuleListIterator;

public: //------------------------------------------------------ public methods

  /// Constructor
  MixedDecimaterT( Mesh& _mesh );

  /// Destructor
  ~MixedDecimaterT();

public:

  /**
   * @brief Decimate (perform _n_collapses collapses). Return number of
   *  performed collapses. If _n_collapses is not given reduce as
   *  much as possible
   * @param _n_collapses Desired number of collapses. If zero (default), attempt
   *                     to do as many collapses as possible.
   * @param _mc_factor   Number between 0 and one defining how much percent of the
   *                     collapses should be performed by MC Decimater before switching to
   *                     Fixed decimation
   * @param _only_selected  Only consider vertices which are selected for decimation
   * @return Number of collapses that were actually performed.
   * @note This operation only marks the removed mesh elements for deletion. In
   *       order to actually remove the decimated elements from the mesh, a
   *       subsequent call to ArrayKernel::garbage_collection() is required.
   */
  size_t decimate( const size_t _n_collapses, const float _mc_factor , bool _only_selected = false);

  /**
   * @brief Decimate the mesh to a desired target vertex complexity.
   * @param _n_vertices Target complexity, i.e. desired number of remaining
   *                    vertices after decimation.

   * @param _only_selected  Only consider vertices which are selected for decimation
   * @param _mc_factor   Number between 0 and one defining how much percent of the
   *                     collapses should be performed by MC Decimater before switching to
   *                     Fixed decimation
   * @return Number of collapses that were actually performed.
   * @note This operation only marks the removed mesh elements for deletion. In
   *       order to actually remove the decimated elements from the mesh, a
   *       subsequent call to ArrayKernel::garbage_collection() is required.
   */
  size_t decimate_to( size_t  _n_vertices, const float _mc_factor , bool _only_selected = false)
  {
    return ( (_n_vertices < this->mesh().n_vertices()) ?
       decimate( this->mesh().n_vertices() - _n_vertices, _mc_factor , _only_selected) : 0 );
  }

  /**
   * @brief Attempts to decimate the mesh until a desired vertex or face
   *        complexity is achieved.
   * @param _n_vertices Target vertex complexity.
   * @param _n_faces Target face complexity.
   * @param _mc_factor   Number between 0 and one defining how much percent of the
   *                     collapses should be performed by MC Decimater before switching to
   *                     Fixed decimation
   * @param _only_selected  Only consider vertices which are selected for decimation
   * @return Number of collapses that were actually performed.
   * @note Decimation stops as soon as either one of the two complexity bounds
   *       is satisfied.
   * @note This operation only marks the removed mesh elements for deletion. In
   *       order to actually remove the decimated elements from the mesh, a
   *       subsequent call to ArrayKernel::garbage_collection() is required.
   */

  size_t decimate_to_faces( const size_t  _n_vertices=0, const size_t _n_faces=0 , const float _mc_factor = 0.8 , bool _only_selected = false);

private: //------------------------------------------------------- private data

};

//=============================================================================
} // END_NS_DECIMATER
} // END_NS_OPENMESH
//=============================================================================
#if defined(OM_INCLUDE_TEMPLATES) && !defined(OPENMESH_MIXED_DECIMATER_DECIMATERT_CC)
#define OPENMESH_MIXED_DECIMATER_TEMPLATES
#include "MixedDecimaterT_impl.hh"
#endif
//=============================================================================
#endif // OPENMESH_MIXED_DECIMATER_DECIMATERT_HH
//=============================================================================
