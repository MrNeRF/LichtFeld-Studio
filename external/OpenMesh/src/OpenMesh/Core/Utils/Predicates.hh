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

#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <OpenMesh/Core/Utils/PropertyManager.hh>

#include <utility>
#include <array>
#include <vector>
#include <set>
#include <type_traits>

//== NAMESPACES ===============================================================

namespace OpenMesh {

namespace Predicates {

//== FORWARD DECLARATION ======================================================

//== CLASS DEFINITION =========================================================

template <typename PredicateT>
struct PredicateBase
{
};

template <typename PredicateT>
struct Predicate : public PredicateBase<Predicate<PredicateT>>
{
  Predicate(PredicateT _p)
    :
      p_(_p)
  {}

  template <typename T>
  bool operator()(const T& _t) const { return p_(_t); }

  PredicateT p_;
};

template <typename PredicateT>
Predicate<const PredicateT&> make_predicate(PredicateT& _p) { return { _p }; }

template <typename PredicateT>
Predicate<PredicateT> make_predicate(PredicateT&& _p) { return { _p }; }

template <typename Predicate1T, typename Predicate2T>
struct Disjunction : public PredicateBase<Disjunction<Predicate1T, Predicate2T>>
{
  Disjunction(Predicate1T _p1, Predicate2T _p2)
    :
      p1_(_p1),
      p2_(_p2)
  {}

  template <typename T>
  bool operator()(const T& _t) const { return p1_( _t) || p2_( _t); }

  Predicate1T p1_;
  Predicate2T p2_;
};

template <typename Predicate1T, typename Predicate2T>
struct Conjunction : public PredicateBase<Conjunction<Predicate1T, Predicate2T>>
{
  Conjunction(Predicate1T _p1, Predicate2T _p2)
    :
      p1_(_p1),
      p2_(_p2)
  {}

  template <typename T>
  bool operator()(const T& _t) const { return p1_( _t) && p2_( _t); }

  Predicate1T p1_;
  Predicate2T p2_;
};


template <typename PredicateT>
struct Negation : public PredicateBase<Negation<PredicateT>>
{
  Negation(const PredicateT& _p1)
    :
      p1_(_p1)
  {}

  template <typename T>
  bool operator()(const T& _t) const { return !p1_( _t); }

  PredicateT p1_;
};

template <typename P1, typename P2>
Disjunction<const P1&, const P2&> operator||(PredicateBase<P1>& p1, PredicateBase<P2>& p2)
{
  return Disjunction<const P1&, const P2&>(static_cast<const P1&>(p1), static_cast<const P2&>(p2));
}

template <typename P1, typename P2>
Disjunction<const P1&, P2> operator||(PredicateBase<P1>& p1, PredicateBase<P2>&& p2)
{
  return Disjunction<const P1&, P2>(static_cast<const P1&>(p1), static_cast<P2&&>(p2));
}

template <typename P1, typename P2>
Disjunction<P1, const P2&> operator||(PredicateBase<P1>&& p1, PredicateBase<P2>& p2)
{
  return Disjunction<P1, const P2&>(static_cast<P1&&>(p1), static_cast<const P2&>(p2));
}

template <typename P1, typename P2>
Disjunction<P1, P2> operator||(PredicateBase<P1>&& p1, PredicateBase<P2>&& p2)
{
  return Disjunction<P1, P2>(static_cast<P1&&>(p1), static_cast<P2&&>(p2));
}

template <typename P1, typename P2>
Conjunction<const P1&, const P2&> operator&&(PredicateBase<P1>& p1, PredicateBase<P2>& p2)
{
  return Conjunction<const P1&, const P2&>(static_cast<const P1&>(p1), static_cast<const P2&>(p2));
}

template <typename P1, typename P2>
Conjunction<const P1&, P2> operator&&(PredicateBase<P1>& p1, PredicateBase<P2>&& p2)
{
  return Conjunction<const P1&, P2>(static_cast<const P1&>(p1), static_cast<P2&&>(p2));
}

template <typename P1, typename P2>
Conjunction<P1, const P2&> operator&&(PredicateBase<P1>&& p1, PredicateBase<P2>& p2)
{
  return Conjunction<P1, const P2&>(static_cast<P1>(p1), static_cast<const P2&>(p2));
}

template <typename P1, typename P2>
Conjunction<P1, P2> operator&&(PredicateBase<P1>&& p1, PredicateBase<P2>&& p2)
{
  return Conjunction<P1, P2>(static_cast<P1&&>(p1), static_cast<P2&&>(p2));
}

template <typename P>
Negation<const P&> operator!(PredicateBase<P>& p)
{
  return Negation<const P&>(static_cast<const P&>(p));
}

template <typename P>
Negation<P> operator!(PredicateBase<P>&& p)
{
  return Negation<P>(static_cast<P&&>(p));
}

struct Feature : public PredicateBase<Feature>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.feature(); }
};

struct Selected : public PredicateBase<Selected>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.selected(); }
};

struct Tagged : public PredicateBase<Tagged>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.tagged(); }
};

struct Tagged2 : public PredicateBase<Tagged2>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.tagged2(); }
};

struct Locked : public PredicateBase<Locked>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.locked(); }
};

struct Hidden : public PredicateBase<Hidden>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.hidden(); }
};

struct Deleted : public PredicateBase<Deleted>
{
  template <typename HandleType>
  bool operator()(const SmartHandleStatusPredicates<HandleType>& _h) const { return _h.deleted(); }
};

struct Boundary : public PredicateBase<Boundary>
{
  template <typename HandleType>
  bool operator()(const SmartHandleBoundaryPredicate<HandleType>& _h) const { return _h.is_boundary(); }
};

template <int inner_reg, int boundary_reg>
struct Regular: public PredicateBase<Regular<inner_reg, boundary_reg>>
{
  bool operator()(const SmartVertexHandle& _vh) const { return _vh.valence() == (_vh.is_boundary() ? boundary_reg : inner_reg); }
};

using RegularQuad = Regular<4,3>;
using RegularTri  = Regular<6,4>;


/// Wrapper object to hold an object and a member function pointer,
/// and provides operator() to call that member function for that object with one argument
template <typename T, typename MF>
struct MemberFunctionWrapper
{
  T t_;    // Objects whose member function we want to call
  MF mf_;  // pointer to member function

  MemberFunctionWrapper(T _t, MF _mf)
    :
      t_(_t),
      mf_(_mf)
  {}

  template <typename O>
  auto operator()(const O& _o) -> decltype ((t_.*mf_)(_o))
  {
    return (t_.*mf_)(_o);
  }
};

/// Helper to create a MemberFunctionWrapper without explicitely naming the types
template <typename T, typename MF>
MemberFunctionWrapper<T,MF> make_member_function_wrapper(T&& _t, MF _mf)
{
  return MemberFunctionWrapper<T,MF>(std::forward<T>(_t), _mf);
}

/// Convenience macro to create a MemberFunctionWrapper for *this object
#define OM_MFW(member_function) OpenMesh::Predicates::make_member_function_wrapper(*this, &std::decay<decltype(*this)>::type::member_function)



//=============================================================================
} // namespace Predicates

} // namespace OpenMesh
//=============================================================================

//=============================================================================
