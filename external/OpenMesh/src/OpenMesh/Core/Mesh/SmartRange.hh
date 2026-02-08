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

#include <utility>
#include <array>
#include <vector>
#include <set>

//== NAMESPACES ===============================================================

namespace OpenMesh {

//== FORWARD DECLARATION ======================================================

//== CLASS DEFINITION =========================================================

namespace  {

struct Identity
{
  template <typename T>
  T operator()(const T& _t) const { return _t; }
};

}

template <typename RangeT, typename HandleT, typename Functor>
struct FilteredSmartRangeT;

/// Base class for all smart range types
template <typename RangeT, typename HandleT>
struct SmartRangeT
{
  using Handle = HandleT;
  using SmartRange = SmartRangeT<RangeT, HandleT>;
  using Range = RangeT;

  // TODO: Someone with better c++ knowledge may improve the code below.
  
  /** @brief Computes number of elements.
   */
  auto size() const -> size_t
  {
      auto range = static_cast<const RangeT*>(this);
      return std::distance(range->begin(), range->end());
  }

  /** @brief Computes the sum of elements.
   *
   * Computes the sum of all elements in the range after applying the functor \p f.
   *
   *  @param f Functor that is applied to all elements before computing the sum
   */
  template <typename Functor>
  auto sum(Functor&& f) const -> typename std::decay<decltype (f(std::declval<HandleT>()))>::type
  {
    auto range = static_cast<const RangeT*>(this);
    auto begin = range->begin();
    auto end   = range->end();
    assert(begin != end);
    typename std::decay<decltype (f(*begin))>::type result = f(*begin);
    auto it = begin;
    ++it;
    for (; it != end; ++it)
        result += f(*it);
    return result;
  }

  /** @brief Computes the average of elements.
   *
   * Computes the average of all elements in the range after applying the functor \p f.
   *
   *  @param f Functor that is applied to all elements before computing the average.
   */
  template <typename Functor>
  auto avg(Functor&& f) const -> typename std::decay<decltype (f(std::declval<HandleT>()))>::type
  {
    auto range = static_cast<const RangeT*>(this);
    auto begin = range->begin();
    auto end   = range->end();
    assert(begin != end);
    typename std::decay<decltype (f(*begin))>::type result = f(*begin);
    auto it = begin;
    ++it;
    int n_elements = 1;
    for (; it != end; ++it)
    {
        result += f(*it);
      ++n_elements;
    }
    return (1.0 / n_elements) * result;
  }

  /** @brief Computes the weighted average of elements.
   *
   * Computes the weighted average of all elements in the range after applying the functor \p f.
   *
   *  @param f Functor that is applied to all elements before computing the average.
   *  @param w Functor returning element weight.
   */
  template <typename Functor, typename WeightFunctor>
  auto avg(Functor&& f, WeightFunctor&& w) const -> typename std::decay<decltype ((1.0/(w(std::declval<HandleT>())+w(std::declval<HandleT>())))*f(std::declval<HandleT>()))>::type
  {
    auto range = static_cast<const RangeT*>(this);
    auto begin = range->begin();
    auto end   = range->end();
    assert(begin != end);
    typename std::decay<decltype (w(*begin))>::type weight = w(*begin);
    typename std::decay<decltype (w(*begin)*f(*begin))>::type result = weight * f(*begin);
    typename std::decay<decltype (w(*begin)+w(*begin))>::type weight_sum = weight;
    auto it = begin;
    ++it;
    for (; it != end; ++it)
    {
      weight = w(*it);
      result += weight*f(*it);
      weight_sum += weight;
    }
    return (1.0 / weight_sum) * result;
  }

  /** @brief Check if any element fulfils condition.
  *
  * Checks if functor \p f returns true for any of the elements in the range.
  * Returns true if that is the case, false otherwise.
  *
  *  @param f Functor that is evaluated for all elements.
  */
  template <typename Functor>
  auto any_of(Functor&& f) const -> bool
  {
    auto range = static_cast<const RangeT*>(this);
    for (auto e : *range)
      if (f(e))
        return true;
    return false;
  }

  /** @brief Check if all elements fulfil condition.
  *
  * Checks if functor \p f returns true for all of the elements in the range.
  * Returns true if that is the case, false otherwise.
  *
  *  @param f Functor that is evaluated for all elements.
  */
  template <typename Functor>
  auto all_of(Functor&& f) const -> bool
  {
    auto range = static_cast<const RangeT*>(this);
    for (auto e : *range)
      if (!f(e))
        return false;
    return true;
  }

  /** @brief Convert range to array.
  *
  * Converts the range of elements into an array of objects returned by functor \p f.
  * The size of the array needs to be provided by the user. If the size is larger than the number of
  * elements in the range, the remaining entries of the array will be uninitialized.
  *
  *  @param f Functor that is applied to all elements before putting them into the array. If no functor is provided
  *           the array will contain the handles.
  */
  template <int n, typename Functor = Identity>
  auto to_array(Functor&& f = {}) const -> std::array<typename std::decay<decltype (f(std::declval<HandleT>()))>::type, n>
  {
    auto range = static_cast<const RangeT*>(this);
    std::array<typename std::decay<decltype (f(std::declval<HandleT>()))>::type, n> res;
    auto it = range->begin();
    auto end = range->end();
    int i = 0;
    while (i < n && it != end)
      res[i++] = f(*(it++));
    return res;
  }

  /** @brief Convert range to vector.
  *
  * Converts the range of elements into a vector of objects returned by functor \p f.
  *
  *  @param f Functor that is applied to all elements before putting them into the vector. If no functor is provided
  *           the vector will contain the handles.
  */
  template <typename Functor = Identity>
  auto to_vector(Functor&& f = {}) const -> std::vector<typename std::decay<decltype (f(std::declval<HandleT>()))>::type>
  {
    auto range = static_cast<const RangeT*>(this);
    std::vector<typename std::decay<decltype (f(std::declval<HandleT>()))>::type> res;
    for (const auto& e : *range)
      res.push_back(f(e));
    return res;
  }

  /** @brief Convert range to set.
  *
  * Converts the range of elements into a set of objects returned by functor \p f.
  *
  *  @param f Functor that is applied to all elements before putting them into the set. If no functor is provided
  *           the set will contain the handles.
  */
  template <typename Functor = Identity>
  auto to_set(Functor&& f = {}) const -> std::set<typename std::decay<decltype (f(std::declval<HandleT>()))>::type>
  {
    auto range = static_cast<const RangeT*>(this);
    std::set<typename std::decay<decltype (f(std::declval<HandleT>()))>::type> res;
    for (const auto& e : *range)
      res.insert(f(e));
    return res;
  }

  /** @brief Get the first element that fulfills a condition.
  *
  * Finds the first element of the range for which the functor \p f evaluates to true.
  * Returns an invalid handle if none evaluates to true
  *
  *  @param f Functor that is applied to all elements before putting them into the set. If no functor is provided
  *           the set will contain the handles.
  */
  template <typename Functor>
  auto first(Functor&& f = {}) const -> HandleT
  {
    auto range = static_cast<const RangeT*>(this);
    for (const auto& e : *range)
      if (f(e))
        return e;
    return HandleT();
  }

  /** @brief Compute minimum.
  *
  * Computes the minimum of all objects returned by functor \p f.
  *
  *  @param f Functor that is applied to all elements before computing minimum.
  */
  template <typename Functor>
  auto min(Functor&& f) const -> typename std::decay<decltype (f(std::declval<HandleT>()))>::type
  {
    using std::min;

    auto range = static_cast<const RangeT*>(this);
    auto it    = range->begin();
    auto end   = range->end();
    assert(it != end);

    typename std::decay<decltype (f(std::declval<HandleT>()))>::type res = f(*it);
    ++it;

    for (; it != end; ++it)
      res = min(res, f(*it));

    return res;
  }

  /** @brief Compute minimal element.
  *
  * Computes the element that minimizes \p f.
  *
  *  @param f Functor that is applied to all elements before comparing.
  */
  template <typename Functor>
  auto argmin(Functor&& f) const -> HandleT
  {
    auto range = static_cast<const RangeT*>(this);
    auto it    = range->begin();
    auto min_it = it;
    auto end   = range->end();
    assert(it != end);

    typename std::decay<decltype (f(std::declval<HandleT>()))>::type curr_min = f(*it);
    ++it;

    for (; it != end; ++it)
    {
      auto val = f(*it);
      if (val < curr_min)
      {
        curr_min = val;
        min_it = it;
      }
    }

    return *min_it;
  }

  /** @brief Compute maximum.
  *
  * Computes the maximum of all objects returned by functor \p f.
  *
  *  @param f Functor that is applied to all elements before computing maximum.
  */
  template <typename Functor>
  auto max(Functor&& f) const -> typename std::decay<decltype (f(std::declval<HandleT>()))>::type
  {
    using std::max;

    auto range = static_cast<const RangeT*>(this);
    auto it    = range->begin();
    auto end   = range->end();
    assert(it != end);

    typename std::decay<decltype (f(std::declval<HandleT>()))>::type res = f(*it);
    ++it;

    for (; it != end; ++it)
      res = max(res, f(*it));

    return res;
  }


  /** @brief Compute maximal element.
  *
  * Computes the element that maximizes \p f.
  *
  *  @param f Functor that is applied to all elements before comparing.
  */
  template <typename Functor>
  auto argmax(Functor&& f) const -> HandleT
  {
    auto range = static_cast<const RangeT*>(this);
    auto it    = range->begin();
    auto max_it = it;
    auto end   = range->end();
    assert(it != end);

    typename std::decay<decltype (f(std::declval<HandleT>()))>::type curr_max = f(*it);
    ++it;

    for (; it != end; ++it)
    {
      auto val = f(*it);
      if (val > curr_max)
      {
        curr_max = val;
        max_it = it;
      }
    }

    return *max_it;
  }

  /** @brief Computes minimum and maximum.
  *
  * Computes the minimum and maximum of all objects returned by functor \p f. Result is returned as std::pair
  * containing minimum as first and maximum as second element.
  *
  *  @param f Functor that is applied to all elements before computing maximum.
  */
  template <typename Functor>
  auto minmax(Functor&& f) const -> std::pair<typename std::decay<decltype (f(std::declval<HandleT>()))>::type,
                                        typename std::decay<decltype (f(std::declval<HandleT>()))>::type>
  {
    return std::make_pair(this->min(f), this->max(f));
  }


  /** @brief Compute number of elements that satisfy a given predicate.
  *
  * Computes the numer of elements which satisfy functor \p f.
  *
  *  @param f Predicate that elements have to satisfy in order to be counted.
  */
  template <typename Functor>
  auto count_if(Functor&& f) const -> int
  {
    int count = 0;
    auto range = static_cast<const RangeT*>(this);
    for (const auto& e : *range)
      if (f(e))
        ++count;
    return count;
  }


  /** @brief Apply a functor to each element.
  *
  * Calls functor \p f with each element as parameter
  *
  *  @param f Functor that is called for each element.
  */
  template <typename Functor>
  auto for_each(Functor&& f) const -> void
  {
    auto range = static_cast<const RangeT*>(this);
    for (const auto& e : *range)
      f(e);
  }


  /** @brief Only iterate over a subset of elements
   *
   * Returns a smart range which skips all elements that do not satisfy functor \p f
   *
   * @param f Functor that needs to be evaluated to true if the element should not be skipped.
   */
  template <typename Functor>
  auto filtered(Functor&& f) const -> FilteredSmartRangeT<SmartRange, Handle, Functor>
  {
    auto range = static_cast<const RangeT*>(this);
    return FilteredSmartRangeT<SmartRange, Handle, Functor>(std::forward<Functor>(f), (*range).begin(), (*range).end());
  }
};


/// Class which applies a filter when iterating over elements
template <typename RangeT, typename HandleT, typename Functor>
struct FilteredSmartRangeT : public SmartRangeT<FilteredSmartRangeT<RangeT, HandleT, Functor>, HandleT>
{
  using BaseRange = SmartRangeT<FilteredSmartRangeT<RangeT, HandleT, Functor>, HandleT>;
  using BaseIterator = decltype((std::declval<typename RangeT::Range>().begin()));

  struct FilteredIterator : public BaseIterator
  {

    FilteredIterator(Functor f, BaseIterator it, BaseIterator end): BaseIterator(it), f_(f), end_(end)
    {
      if (!BaseIterator::operator==(end_) && !f_(*(*this))) // if start is not valid go to first valid one
        operator++();
    }

    FilteredIterator(const FilteredIterator& other) = default;

    FilteredIterator& operator=(const FilteredIterator& other)
    {
      BaseIterator::operator=(other);
      end_ = other.end_;
      return *this;
    }

    FilteredIterator& operator++()
    {
      if (BaseIterator::operator==(end_)) // don't go past end
        return *this;

      // go to next valid one
      do
        BaseIterator::operator++();
      while (BaseIterator::operator!=(end_) && !f_(*(*this)));
      return *this;
    }

    Functor f_;        // Should iterators always get a reference to filter stored in range?
                       // Should iterators stay valid after range goes out of scope?
    BaseIterator end_;
  };

  FilteredSmartRangeT(Functor&& f, BaseIterator begin, BaseIterator end) : f_(std::forward<Functor>(f)), begin_(std::move(begin)), end_(std::move(end)){}
  FilteredIterator begin() const { return FilteredIterator(f_, begin_, end_); }
  FilteredIterator end()   const { return FilteredIterator(f_, end_, end_); }

  Functor f_;
  BaseIterator begin_;
  BaseIterator end_;
};



//=============================================================================
} // namespace OpenMesh
//=============================================================================

//=============================================================================
