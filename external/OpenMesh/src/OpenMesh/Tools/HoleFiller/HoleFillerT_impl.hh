/* ========================================================================= *
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2023, RWTH-Aachen University                 *
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



//=============================================================================
#include "HoleFillerT.hh"
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>
//=============================================================================

//== NAMESPACES ===============================================================


namespace OpenMesh {
namespace HoleFiller {

template< class MeshT >
HoleFillerT< MeshT >::HoleFillerT(MeshT &_mesh )
  : mesh_( _mesh )
{
  mesh_.request_vertex_status();
  mesh_.request_edge_status();

  if (! mesh_.get_property_handle(scale_,"scale") )
    mesh_.add_property( scale_ , "scale" );
}


//=============================================================================



template< class MeshT >
HoleFillerT< MeshT >::~HoleFillerT()
{
  mesh_.release_vertex_status();
  mesh_.release_edge_status();

  if ( mesh_.get_property_handle(scale_,"scale") )
    mesh_.remove_property( scale_ );
}


//=============================================================================
//
// Identify and fill all holes of the mesh.
//
//=============================================================================


template< class MeshT >
void
HoleFillerT< MeshT >::fill_all_holes( int _stages )
{


  // Collect all boundary edges  
  std::vector< typename MeshT::EdgeHandle > bdry_edge;
  
  for (auto ei : mesh_.edges())
    if ( ei.is_boundary() )
      bdry_edge.push_back( ei );
  

  // Fill holes
  int cnt = 0;
  for (auto i : bdry_edge)
    if ( mesh_.is_boundary( i ) )
    {
      ++cnt;
      omlog() << "Filling hole " << cnt << "\n";
      fill_hole( i, _stages );
    }
  
  // Smooth fillings
  if ( _stages <= 2 )
    return;

  omlog() << "Stage 3 : Smoothing the hole fillings ... ";
  
  OpenMesh::Smoother::JacobiLaplaceSmootherT< MeshT > smoother( mesh_ );
  smoother.initialize( OpenMesh::Smoother::SmootherT< MeshT >::
		       Tangential_and_Normal,
               OpenMesh::Smoother::SmootherT< MeshT >::C1 );
  
  smoother.smooth( 500 );
}


//=============================================================================
//
// Fill a hole which is identified by one of its boundary edges.
//
//=============================================================================


template< class MeshT >
void
HoleFillerT< MeshT >::fill_hole(typename MeshT::EdgeHandle _eh, int _stages )
{

  omlog() << "  Stage 1 : Computing a minimal triangulation ... ";

  // remember last vertex for selection of new ones
  typename MeshT::VertexHandle old_last_handle = *(--mesh_.vertices_end());

  // No boundary edge, no hole
  if ( ! mesh_.is_boundary( _eh ) ) {
      omerr() << "fill_hole: Given edge handle is not a boundary edge at a hole!" << std::endl;
    return;
  }
  
  // Get boundary halfedge  
  OpenMesh::SmartHalfedgeHandle hh = make_smart(_eh, mesh_).h0();

  if ( ! hh.is_boundary() )
    hh = hh.opp();
  
  // Collect boundary vertices
  boundary_vertex_.clear();
  opposite_vertex_.clear();
  
  OpenMesh::SmartHalfedgeHandle ch = hh;

  do {
    boundary_vertex_.push_back( ch.from() );
    opposite_vertex_.push_back( ch.opp().next().to() );
    //check number of outgoing boundary HEH's at Vertex
    int c = 0;
    OpenMesh::SmartVertexHandle vh = ch.to();

    for (auto voh_it : vh.outgoing_halfedges())
      if ( voh_it.is_boundary() )
        c++;

    if ( c >= 2){
      OpenMesh::SmartHalfedgeHandle op = ch.opp();
      typename MeshT::VertexOHalfedgeIter voh_it(mesh_,op);

      ch = *(++voh_it);
    }else
      ch = ch.next();

  } while ( ch != hh );

  
  int nv = boundary_vertex_.size();
  
  // Compute an initial triangulation 
  w_.clear();
  w_.resize( nv, std::vector<Weight>( nv, Weight() ) );
  
  l_.clear();
  l_.resize( nv, std::vector<int>( nv, 0 ) );
  

  for ( int i = 0; i < nv - 1; ++i )
    w_[i][i+1] = Weight( 0, 0 );
  
  for ( int j = 2; j < nv; ++j )
  {
    #pragma omp parallel for shared(j, nv)
    for(int i = 0; i < nv-j; ++i)
    {
      Weight valmin;
      int   argmin = -1;
      for ( int m = i + 1; m < i + j; ++m )
      {
	Weight newval = w_[i][m] + w_[m][i+j] + weight( i, m, i+j );
	if ( newval < valmin )
	{
	  valmin = newval;
	  argmin = m;
	}
      }
      
      w_[i][i+j] = valmin;
      l_[i][i+j] = argmin;
    }
  }

  // Actually fill the hole. We collect all triangles and edges of
  // this filling for further processing. 
  hole_edge_.clear();
  hole_triangle_.clear();
  if ( fill( 0, nv - 1 ) ){

    if ( _stages <= 1 )
      return;
    
    omlog() << "  Stage 2 : Fairing the filling ... " << std::endl;
    
    std::vector< OpenMesh::SmartFaceHandle > handles = hole_triangle_;
    
    fairing(handles);

    //select all new vertices
    typename MeshT::VertexIter old_end = ++typename MeshT::VertexIter(mesh_,old_last_handle);
    typename MeshT::VertexIter v_end = mesh_.vertices_end();

    for(; old_end != v_end; ++old_end)
      if ( !mesh_.status(*old_end).deleted() )
        mesh_.status(*old_end).set_selected( true );

  }else
      omerr() << "Could not create triangulation" << std::endl;
}


/// path fairing
template< class MeshT >
void
HoleFillerT< MeshT >::fairing( std::vector< OpenMesh::SmartFaceHandle >& _faceHandles ){

  //generate vector of all edges
  hole_edge_.clear();

  hole_triangle_ = _faceHandles;

  OpenMesh::EPropHandleT< bool > edgeProp;
  OpenMesh::VPropHandleT< bool > vertexProp;
  OpenMesh::FPropHandleT< bool > faceProp;
  OpenMesh::FPropHandleT< bool > orderProp;

  if (! mesh_.get_property_handle(edgeProp,"edgeProp") )
    mesh_.add_property( edgeProp, "edgeProp" );
  if (! mesh_.get_property_handle(vertexProp,"vertexProp") )
    mesh_.add_property( vertexProp, "vertexProp" ); 
  if (! mesh_.get_property_handle(faceProp,"faceProp") )
    mesh_.add_property( faceProp, "faceProp" ); 
  if (! mesh_.get_property_handle(orderProp,"orderProp") )
    mesh_.add_property( orderProp, "orderProp" ); 

  //init properties by setting all of them to false
  for (auto fIt : mesh_.faces()) {
    mesh_.property( orderProp, fIt ) = false;
    mesh_.property( faceProp, fIt ) = false;
  }

  for (auto eIt : mesh_.edges())
    mesh_.property( edgeProp, eIt ) = false;

  for (auto vIt : mesh_.vertices()) {
    mesh_.property( vertexProp, vIt ) = false;
  }

  //set face property
  for (uint i = 0; i < hole_triangle_.size(); i++){
    mesh_.property( faceProp, hole_triangle_[i] ) = true;
  }

  //set properties
  for (unsigned int i = 0; i < hole_triangle_.size(); i++){
      for (auto fei : hole_triangle_[i].edges()) {
      mesh_.status( fei ).set_locked(true);
      //set edge property for all edges inside the hole (eg not on the hole boundary)
      if (mesh_.property( faceProp, fei.h0().face() ) &&
          mesh_.property( faceProp, fei.h1().face() ) ){

        mesh_.property( edgeProp, fei ) = true;
        hole_edge_.push_back( fei );
        mesh_.status( fei ).set_locked(false);
      }
    }

    /// @TODO, strange iterator at property!
    for (auto fvi : hole_triangle_[i].vertices()){
      //set vertex property for all vertices of the hole
      for ( auto vfi : fvi.faces() )
        mesh_.property( vertexProp, fvi ) = true;
    }
  }

  //calculate scaling weights for vertices
  for (auto vIt : mesh_.vertices())
    if (mesh_.property(vertexProp, vIt)){

      Scalar cnt   = 0;
      Scalar scale = 0;
      
      for ( auto voh_it : vIt.outgoing_halfedges())
      {

        if (voh_it.face().is_valid() &&
            voh_it.opp().face().is_valid() &&
            mesh_.property(faceProp, voh_it.face() ) &&
            mesh_.property(faceProp, voh_it.opp().face() ))
          continue;

        cnt += 1.0f;
        Point p0 = mesh_.point( vIt );
        Point p1 = mesh_.point( voh_it.to() );
        scale += norm( p1 - p0 );

      }

      scale /= cnt;
      
      mesh_.property( scale_, vIt ) = scale;
    }

  mesh_.remove_property(edgeProp);
  mesh_.remove_property(vertexProp);
  mesh_.remove_property(faceProp);
  mesh_.remove_property(orderProp);

  // Do the patch fairing
  bool did_refine = true;

  for ( int k = 0; k < 40 && did_refine; ++k )
  {
    uint end = hole_triangle_.size();

    did_refine = false;
    for ( unsigned int j = 0; j < end; ++j )
        did_refine |= refine( hole_triangle_[j] );

    for ( int i = 0; i < 10; ++i )
      for ( unsigned int j = 0; j < hole_edge_.size(); ++j )
        relax_edge( hole_edge_[j] );
  }

  // unlock everything
  for ( auto ei : mesh_.edges())
    mesh_.status( ei ).set_locked( false );
}

//=============================================================================
//
// Refine a face: Apply a 1-to-3 split if the edge lengths of the
// face do not match the interpolated edge lengths of the hole
// boundary.
//
//=============================================================================


template< class MeshT >
bool
HoleFillerT< MeshT >::refine(typename MeshT::FaceHandle _fh )
{
 
  // Collect the three edges of the face into e0, e1, e2

  typename MeshT::FEIter fei = mesh_.fe_iter( _fh );
  OpenMesh::SmartEdgeHandle  e0  = *fei; ++fei;
  OpenMesh::SmartEdgeHandle  e1  = *fei; ++fei;
  OpenMesh::SmartEdgeHandle  e2  = *fei; ++fei;


  // Collect the vertices, vertex positions and scale factors of the face.

  typename MeshT::FVIter   fvi = mesh_.fv_iter( _fh );

  typename MeshT::VertexHandle    v0 = *fvi; ++fvi;
  typename MeshT::VertexHandle    v1 = *fvi; ++fvi;
  typename MeshT::VertexHandle    v2 = *fvi; ++fvi;

  Point p0 = mesh_.point( v0 );
  Point p1 = mesh_.point( v1 );
  Point p2 = mesh_.point( v2 );

  Scalar scale0 = mesh_.property( scale_, v0 );
  Scalar scale1 = mesh_.property( scale_, v1 );
  Scalar scale2 = mesh_.property( scale_, v2 );

  // Interpolate the scale factor.

  Scalar scale = ( scale0 + scale1 + scale2 ) / 3.0f;
  Point center = typename MeshT::Scalar(1.0/3.0) * ( p0 + p1 + p2 );

  Scalar d0 = 1.0f * norm( p0 - center );
  Scalar d1 = 1.0f * norm( p1 - center );
  Scalar d2 = 1.0f * norm( p2 - center );


  //dont split triangles which tend to degenerate
  if ( (d0 + d1 + d2) / 3.0f < scale) return false;


  // If the edge lengths differ too much from the scale, perform a
  // triangle split.

  if ( ( d0 > scale && d0 > scale0 ) ||
       ( d1 > scale && d1 > scale1 ) ||
       ( d2 > scale && d2 > scale2 ) )
  {
    // Split the face ...
    OpenMesh::SmartVertexHandle ch = mesh_.add_vertex( center );
    mesh_.split( _fh, ch );

    // ... put the new triangles into the global triangle list ...

    for ( auto vfi : ch.faces() )
      if ( vfi != _fh )
         hole_triangle_.push_back( vfi );

    // ... put the new edges into the global edge list ...

    for ( auto vei : ch.edges() )
      hole_edge_.push_back( vei );

    // ... and set the appropriate scale factor for the new vertex.

    mesh_.property( scale_, ch ) = scale;

    relax_edge( e0 );
    relax_edge( e1 );
    relax_edge( e2 );

    return true;
  }

  return false;
}


//=============================================================================
//
// Relax an edge: Flip it if one of its opposing vertices lies in
// the circumsphere of the other three vertices.
//
//=============================================================================


template< class MeshT >
bool
HoleFillerT< MeshT >::relax_edge( OpenMesh::SmartEdgeHandle _eh )
{
  if ( mesh_.status( _eh ).locked() )
   return false;

  // Abbreviations for the two halfedges of _eh

  OpenMesh::SmartHalfedgeHandle h0 = _eh.h0();
  OpenMesh::SmartHalfedgeHandle h1 = _eh.h1();

  // Get the two end-vertices u and v of the edge
	 
  Point u( mesh_.point( h0.to() ) );
  Point v( mesh_.point( h1.to() ) );

  // Get the two opposing vertices a and b
    
  Point a( mesh_.point( h0.next().to() ) );
  Point b( mesh_.point( h1.next().to() ) );

  // If the circumsphere criterion is fullfilled AND if the flip is
  // topologically admissible, we do it.

  if ( in_circumsphere( a, u, v, b ) || in_circumsphere( b, u, v, a ) ){
    if ( mesh_.is_flip_ok( _eh ) )
    {

      mesh_.flip( _eh );

      return true;
    }else
      mesh_.status(_eh).set_selected( true );
}
  return false;
}


//=============================================================================
//
// Test whether a point _x lies in the circumsphere of _a,_b,_c.
//
//=============================================================================


template< class MeshT >
bool
HoleFillerT< MeshT >::in_circumsphere( const Point & _x,
				    const Point & _a,
				    const Point & _b,
				    const Point & _c ) const
{
  Point ab = _b - _a;
  Point ac = _c - _a;

  Scalar a00 = -2.0f * ( dot(ab , _a ) );
  Scalar a01 = -2.0f * ( dot(ab , _b ) );
  Scalar a02 = -2.0f * ( dot(ab , _c ) );
  Scalar b0 = norm(_a)*norm(_a) - norm(_b)*norm(_b);

  Scalar a10 = -2.0f * ( dot(ac , _a ) );
  Scalar a11 = -2.0f * ( dot(ac , _b ) );
  Scalar a12 = -2.0f * ( dot(ac , _c ) );
  Scalar b1 = norm(_a)*norm(_a) - norm(_c)*norm(_c);

  typename MeshT::Scalar alpha = -(-a11*a02+a01*a12-a12*b0+b1*a02+a11*b0-a01*b1)
	                             / (-a11*a00+a11*a02-a10*a02+a00*a12+a01*a10-a01*a12);
  typename MeshT::Scalar beta  =  (a10*b0-a10*a02-a12*b0+a00*a12+b1*a02-a00*b1)
	                             / (-a11*a00+a11*a02-a10*a02+a00*a12+a01*a10-a01*a12);
  typename MeshT::Scalar gamma =  (-a11*a00-a10*b0+a00*b1+a11*b0+a01*a10-a01*b1)
                                / (-a11*a00+a11*a02-a10*a02+a00*a12+a01*a10-a01*a12);

  Point center = alpha * _a + beta * _b + gamma * _c;

  return norm( _x - center ) * norm( _x - center ) < norm( _a - center ) * norm( _a - center );
}


//=============================================================================
//
// Create the triangulation
//
// Recursively creates the triangulation for polygon (_i,...,_j).
//
//=============================================================================


template< class MeshT >			
bool
HoleFillerT< MeshT >::fill( int _i, int _j )
{
  // If the two vertices _i and _j are adjacent, there is nothing to do.

  if ( _i + 1 == _j )
    return true;
    

  // Create and store the middle triangle, store its edges.
    
  OpenMesh::SmartFaceHandle fh = mesh_.add_face( boundary_vertex_[_i],
			  boundary_vertex_[ l_[_i][_j] ],
			  boundary_vertex_[_j] );
  hole_triangle_.push_back( fh );

  if (!fh.is_valid())
    return false;

  hole_edge_.push_back( mesh_.edge_handle
			( mesh_.find_halfedge( boundary_vertex_[_i],
					       boundary_vertex_[ l_[_i][_j] ] ) ) );
  hole_edge_.push_back( mesh_.edge_handle
			( mesh_.find_halfedge( boundary_vertex_[ l_[_i][_j] ],
					       boundary_vertex_[_j] ) ) );
  
  
  // Recursively create the left and right side of the
  // triangulation.
    
  if (!fill( _i, l_[_i][_j] ) || !fill( l_[_i][_j], _j ))
    return false;
  else 
    return true;
}



//=============================================================================
//
// Compute the weight of the triangle (_i,_j,_k).
//
//=============================================================================


template< class MeshT >			
typename HoleFillerT< MeshT >::Weight
HoleFillerT< MeshT >::weight( int _i, int _j, int _k )
{
  // Return an infinite weight if the insertion of this triangle
  // would create complex edges.

  if ( exists_edge( boundary_vertex_[_i], boundary_vertex_[_j] ) ||
       exists_edge( boundary_vertex_[_j], boundary_vertex_[_k] ) ||
       exists_edge( boundary_vertex_[_k], boundary_vertex_[_i] ) )
    return Weight();


  // Return an infinite weight, if one of the neighboring patches
  // could not be created.

    
  if ( l_[_i][_j] == -1 ) return Weight();
  if ( l_[_j][_k] == -1 ) return Weight();

   
  // Compute the maxmimum dihedral angles to the adjacent triangles
  // (if they exist)

  Scalar angle = 0.0f;

  if ( _i + 1 == _j )
    angle = std::max( angle, dihedral_angle( boundary_vertex_[_i],
					     boundary_vertex_[_j],
					     boundary_vertex_[_k],
					     opposite_vertex_[_i] ) );
  else
    angle = std::max( angle, dihedral_angle( boundary_vertex_[_i],
					     boundary_vertex_[_j],
					     boundary_vertex_[_k],
					     boundary_vertex_[l_[_i][_j]] ) );
    
  if ( _j + 1 == _k )
    angle = std::max( angle, dihedral_angle( boundary_vertex_[_j],
					     boundary_vertex_[_k],
					     boundary_vertex_[_i],
					     opposite_vertex_[_j] ) );
  else
    angle = std::max( angle, dihedral_angle( boundary_vertex_[_j],
					     boundary_vertex_[_k],
					     boundary_vertex_[_i],
					     boundary_vertex_[l_[_j][_k]] ) );
    
  if ( _i == 0 && _k == (int) boundary_vertex_.size() - 1 )
    angle = std::max( angle, dihedral_angle( boundary_vertex_[_k],
					     boundary_vertex_[_i],
					     boundary_vertex_[_j],
					     opposite_vertex_[_k] ) );


  return Weight( angle, area( boundary_vertex_[_i],
			      boundary_vertex_[_j],
			      boundary_vertex_[_k] ) );
}



//=============================================================================
//
// Does an edge from vertex _u to _v exist?
//
//=============================================================================


template< class MeshT >			
bool
HoleFillerT< MeshT >::exists_edge( OpenMesh::SmartVertexHandle _u, typename MeshT::VertexHandle _w )
{
  for ( auto vohi : _u.outgoing_halfedges() )
    if ( ! vohi.edge().is_boundary() )
      if ( vohi.to() == _w )
	return true;

  return false;
}



//=============================================================================
//
// Compute the area of the triangle (_a,_b,_c).
//
//=============================================================================


template< class MeshT >			
typename MeshT::Scalar
HoleFillerT< MeshT >::area( typename MeshT::VertexHandle _a, typename MeshT::VertexHandle _b, typename MeshT::VertexHandle _c )
{
  Point a( mesh_.point( _a ) );
  Point b( mesh_.point( _b ) );
  Point c( mesh_.point( _c ) );

  Point n( cross(( b - a ) , ( c - b )) );

  return 0.5 * norm(n);
}


//=============================================================================
//
// Compute a dihedral angle
//
// Computes the dihedral angle (in degrees) between triangle
// (_u,_v,_a) and triangle (_v,_u,_b), no matter whether these
// triangles actually exist in the current mesh or not).
//
//=============================================================================


template< class MeshT >			
typename MeshT::Scalar
HoleFillerT< MeshT >::dihedral_angle( typename MeshT::VertexHandle _u, typename MeshT::VertexHandle _v, typename MeshT::VertexHandle _a, typename MeshT::VertexHandle _b )
{
  Point u( mesh_.point( _u ) );
  Point v( mesh_.point( _v ) );
  Point a( mesh_.point( _a ) );
  Point b( mesh_.point( _b ) );

  Point n0( cross(( v - u ) , ( a - v )) );
  Point n1( cross(( u - v ) , ( b - u )) );
    
  normalize(n0);
  normalize(n1);

  return acos( dot(n0,n1) ) * 180.0 / M_PI;
}


/// remove degenerated faces
template< class MeshT >
void
HoleFillerT< MeshT >::removeDegeneratedFaces( std::vector< typename MeshT::FaceHandle >& _faceHandles ){

  for (int i = _faceHandles.size()-1; i >= 0 ; i--){

    if ( mesh_.status( _faceHandles[i] ).deleted() ){
      // face might be deleted because of a previous edge collapse
      // erase the face from the vector
      _faceHandles.erase( _faceHandles.begin() + i );
      continue;
    }

    //get the vertices (works only on triMeshes)
    typename MeshT::FaceVertexIterator fvi = mesh_.fv_iter( _faceHandles[i] );
    Point v0 = mesh_.point( *fvi);
    ++fvi;
    Point v1 = mesh_.point( *fvi );
    ++fvi;
    Point v2 = mesh_.point( *fvi );

    //check if its degenerated
    Point v0v1 = v1 - v0;
    Point v0v2 = v2 - v0;
    Point n = v0v1 % v0v2; // not normalized !
    double d = n.sqrnorm();

    if (d < FLT_MIN && d > -FLT_MIN) {
      // degenerated face found
      typename MeshT::FaceHalfedgeIterator hIt = mesh_.fh_iter( _faceHandles[i] );

      //try to collapse one of the edges
      while (hIt.is_valid()){
        if ( mesh_.is_collapse_ok( *hIt ) ){
          // collapse the edge to remove the triangle
          mesh_.collapse( *hIt );
          // and erase the corresponding face from the vector
          _faceHandles.erase( _faceHandles.begin() + i );
          break;
        } else {
          ++hIt;
        }
      }
    }
  }
}

//=============================================================================


//=============================================================================
} // namespace HoleFiller
} // namespace OpenMesh
//=============================================================================
