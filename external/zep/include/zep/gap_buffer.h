/*The MIT License (MIT)

Copyright (c) 2017 chrismaughan.com

https://github.com/cmaughan/
https://chrismaughan.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*/

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>

#ifdef _DEBUG
#define DEBUG_FILL_GAP                                      \
    for (auto* pCh = m_pGapStart; pCh < m_pGapEnd; pCh++) { \
        *pCh = '@';                                         \
    }
#else
#define DEBUG_FILL_GAP
#endif

// An STL-friendly GapBuffer
// This is my attempt at a Gap buffer which looks a bit like a vector
// The idea is that to the client, it looks like contiguous memory, but internally there is a gap
// Any modification operations may involve moving (and possibly) resizing the gap
// A Gap buffer is a special type of buffer that has favourable performance when inserting/removing entries
// in a local region.  Thus memory moving cost is amortised
// Editors like emacs use it to efficiently manage an edit buffer.
// For the curious, you can ask emacs for the gap position and fixed_size by evaluating (gap-fixed_size), (gap-position)

template <class T, class A = std::allocator<T>>
class GapBuffer {
public:
    static const int DEFAULT_GAP = 1000;

    typedef A allocator_type;
    typedef typename std::allocator_traits<A>::value_type value_type;
    typedef typename std::allocator_traits<A>::difference_type difference_type;
    typedef typename std::allocator_traits<A>::size_type size_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    T* m_pStart = nullptr;    // Start of the buffer
    T* m_pEnd = nullptr;      // Pointer after the end
    T* m_pGapStart = nullptr; // Gap start position
    T* m_pGapEnd = nullptr;   // End of the gap, just beyond
    size_t m_defaultGap = 0;  // Current fixed_size of gap
    A _alloc;                 // The memory allocator to use

    // An iterator used to walk the buffer
    class iterator {
    public:
        typedef typename std::allocator_traits<A>::difference_type difference_type;
        typedef typename std::allocator_traits<A>::value_type value_type;
        typedef typename std::allocator_traits<A>::pointer pointer;
        typedef std::random_access_iterator_tag iterator_category; // or another tag
        typedef value_type& reference;

        bool skipGap = true;
        // Note: p is measured in actual characters, not entire buffer fixed_size.
        // i.e. p skips the gap, and is always less than fixed_size()!
        size_t p = 0;

        GapBuffer<T>& buffer;

        iterator(GapBuffer<T>& buff, size_t ptr, bool skip = true) : skipGap(skip), p(ptr), buffer(buff) {}
        iterator(const iterator& rhs) : skipGap(rhs.skipGap), p(rhs.p), buffer(rhs.buffer) {}
        ~iterator() {}

        iterator& operator=(const iterator& rhs) {
            p = rhs.p;
            return *this;
        }

        bool operator==(const iterator& rhs) const { return (p == rhs.p); }
        bool operator!=(const iterator& rhs) const { return (p != rhs.p); }

        bool operator<(const iterator& rhs) const { return (p < rhs.p); }
        bool operator>(const iterator& rhs) const { return (p > rhs.p); }
        bool operator<=(const iterator& rhs) const { return (p <= rhs.p); }
        bool operator>=(const iterator& rhs) const { return (p >= rhs.p); }

        iterator& operator+=(size_type rhs) {
            p += rhs;
            return *this;
        }
        iterator operator+(size_type rhs) const { return iterator(buffer, p + rhs); }
        friend iterator operator+(size_type lhs, const iterator& rhs) { return iterator(lhs.buffer, lhs + rhs.p); }
        iterator& operator-=(size_type rhs) {
            p -= rhs;
            return *this;
        }
        iterator operator-(size_type rhs) const { return iterator(buffer, p - rhs); }
        difference_type operator-(iterator itr) const { return p - itr.p; }

        iterator& operator--() {
            p--;
            return *this;
        }
        iterator operator--(int) {
            auto pOld = p;
            p--;
            return iterator(buffer, pOld);
        }
        iterator& operator++() {
            p++;
            return *this;
        }
        iterator operator++(int) {
            auto pOld = p;
            p++;
            return iterator(buffer, pOld);
        }

        reference operator*() const { return *buffer.GetBufferPtr(p, skipGap); }
        pointer operator->() const { return buffer.GetBufferPtr(p, skipGap); }
        reference operator[](size_type distance) const { return buffer.GetBufferPtr(p + distance, skipGap); }
    };

    class const_iterator {
    public:
        typedef typename std::allocator_traits<A>::difference_type difference_type;
        typedef typename std::allocator_traits<A>::value_type value_type;
        typedef typename std::allocator_traits<A>::pointer pointer;
        typedef std::random_access_iterator_tag iterator_category; // or another tag
        typedef value_type& reference;

        bool skipGap = true;
        size_t p = 0;
        const GapBuffer<T>& buffer;

        const_iterator(const GapBuffer<T>& buff, size_t ptr, bool skip = true) : skipGap(skip), p(ptr), buffer(buff) {}
        const_iterator(const iterator& rhs) : skipGap(rhs.skipGap), p(rhs.p), buffer(rhs.buffer) {}
        const_iterator(const const_iterator& rhs) : skipGap(rhs.skipGap), p(rhs.p), buffer(rhs.buffer) {}
        ~const_iterator() {}

        const_iterator& operator=(const const_iterator& rhs) {
            p = rhs.p;
            return *this;
        }
        bool operator==(const const_iterator& rhs) const { return (p == rhs.p); }
        bool operator!=(const const_iterator& rhs) const { return (p != rhs.p); }

        bool operator<(const const_iterator& rhs) const { return (p < rhs.p); }
        bool operator>(const const_iterator& rhs) const { return (p > rhs.p); }
        bool operator<=(const const_iterator& rhs) const { return (p <= rhs.p); }
        bool operator>=(const const_iterator& rhs) const { return (p >= rhs.p); }

        const_iterator& operator++() {
            p++;
            return *this;
        };
        const_iterator operator++(int) {
            auto pOld = p;
            p++;
            return const_iterator(buffer, pOld);
        }
        const_iterator& operator--() {
            p--;
            return *this;
        }
        const_iterator operator--(int) {
            auto pOld = p;
            p--;
            return const_iterator(buffer, pOld);
        }

        const_iterator& operator+=(size_type rhs) {
            p += rhs;
            return *this;
        }
        const_iterator operator+(size_type rhs) const { return const_iterator(buffer, p + rhs); }
        friend const_iterator operator+(size_type lhs, const const_iterator& rhs) { return const_iterator(rhs.buffer, lhs + rhs.p); }
        const_iterator& operator-=(size_type rhs) {
            p -= rhs;
            return *this;
        }
        const_iterator operator-(size_type rhs) const { return const_iterator(buffer, p - rhs); }
        difference_type operator-(const_iterator itr) const { return p - itr.p; }

        reference operator*() const { return *buffer.GetBufferPtr(p, skipGap); }
        pointer operator->() const { return buffer.GetBufferPtr(p, skipGap); }
        reference operator[](size_type distance) const { return buffer.GetBufferPtr(p + distance, skipGap); }
    };

    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

public:
    iterator begin() { return iterator(*this, 0); }
    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator cbegin() const { return const_iterator(*this, 0); }
    iterator end() { return iterator(*this, size()); };
    const_iterator end() const { return const_iterator(*this, size()); }
    const_iterator cend() const { return const_iterator(*this, size()); }

    bool operator==(const GapBuffer<T>& rhs) const { return this == &rhs; }
    bool operator!=(const GapBuffer<T>& rhs) const { return this != &rhs; }

    // Size is fixed_size of buffer - the gap fixed_size
    inline size_type size() const { return (m_pEnd - m_pStart) - (m_pGapEnd - m_pGapStart); }

    // No current limit ; not sure how to calculate the ram max fixed_size here
    size_type max_size() const { return std::numeric_limits<size_t>::max(); }

    // An empty buffer
    inline bool empty() const { return size() == 0; }

    // Return the allocator
    A get_allocator() const { return _alloc; }

    // No assign/copy for now
    GapBuffer(int size = 0, int gapSize = DEFAULT_GAP)
        : m_defaultGap(gapSize) {
        if (size == 0) {
            clear();
        } else {
            resize(size);
        }
    }

    // No copy constructor yet
    GapBuffer(const GapBuffer& copy) = delete;
    GapBuffer& operator=(const GapBuffer& copy) = delete;

    virtual ~GapBuffer() { Free(); }

    // Make an empty buffer, with just the gap in it
    // It appears 'empty' even though there is gap space that can be filled
    void clear() {
        Free();

        m_pStart = get_allocator().allocate(m_defaultGap);

        //  SG_____gE -- gap visualization
        m_pGapStart = m_pStart;
        m_pGapEnd = m_pStart + m_defaultGap;
        m_pEnd = m_pGapEnd;
    }

    // Make buffer this fixed_size, but only ever actually grow the memory for now.
    void resize(size_t newSize) {
        auto sizeIncrease = (int64_t)newSize - (int64_t)size();
        if (sizeIncrease == 0) {
            return;
        }

        // If the buffer is shrunk, we move the gap to the end and make it bigger
        // Thus we don't make the buffer smaller, we make the gap bigger instead.
        // From the outside the fixed_size() is the smaller fixed_size.
        if (sizeIncrease < 0) {
            // Move the gap to the end
            MoveGap(size());
            m_pGapStart += sizeIncrease;
            return;
        }

        // New total fixed_size and new gap fixed_size
        auto bufferSize = CurrentSizeWithGap() + sizeIncrease;

        // Make the new buffer
        auto pNewStart = get_allocator().allocate((const size_t)bufferSize);

        memcpy(pNewStart, m_pStart, CurrentSizeWithGap() * sizeof(T));

        // First section, before gap - copy into place
        if ((m_pGapStart - m_pStart) > 0) {
            memcpy(pNewStart, m_pStart, (m_pGapStart - m_pStart) * sizeof(T));
        }

        // Free the old memory
        if (m_pStart) {
            get_allocator().deallocate(m_pStart, CurrentSizeWithGap());
        }

        // Fix up the new pointers (gap hasn't changed, end is bigger)
        auto gapSize = CurrentGapSize();
        m_pGapStart = pNewStart + (m_pGapStart - m_pStart);
        m_pGapEnd = m_pGapStart + gapSize;
        m_pStart = pNewStart;
        m_pEnd = pNewStart + bufferSize;
    }

    // Resize the gap to this fixed_size
    void resizeGap(size_t newGapSize) {
        auto sizeIncrease = int64_t(newGapSize) - int64_t(CurrentGapSize());
        if (sizeIncrease <= 0) {
            // We never shrink the gap
            return;
        }

        // New total fixed_size and new gap fixed_size
        auto bufferSize = CurrentSizeWithGap() + sizeIncrease;

        // Make the new buffer
        auto pNewStart = get_allocator().allocate((const size_t)bufferSize);

        // First section, before gap - copy into place
        if ((m_pGapStart - m_pStart) > 0) {
            memcpy(pNewStart, m_pStart, (m_pGapStart - m_pStart) * sizeof(T));
        }

        // Note: Gap is copied; it is 'junk until used'

        // Last section - copy into place
        auto pNewGapStart = m_pGapStart + (pNewStart - m_pStart);
        auto pNewGapEnd = pNewGapStart + newGapSize;
        if (m_pGapEnd < m_pEnd) {
            memcpy(pNewGapEnd, m_pGapEnd, (m_pEnd - m_pGapEnd) * sizeof(T));
        }

        // Free the old memory
        if (m_pStart) {
            get_allocator().deallocate(m_pStart, CurrentSizeWithGap());
        }

        // Fix up the new pointers
        m_pGapStart = pNewStart + (m_pGapStart - m_pStart);
        m_pGapEnd = m_pGapStart + newGapSize;
        m_pStart = pNewStart;
        m_pEnd = pNewStart + bufferSize;

        DEBUG_FILL_GAP;
    }

    // Return a string version of the gap, optionally showing the gap fixed_size
    // inside ||.  This is used for testing/validation
    std::string string(bool showGap = false) const {
        std::string str;

        if (m_pGapStart - m_pStart) {
            str.append((const char*)m_pStart, (size_type)(m_pGapStart - m_pStart));
        }

        if (showGap) {
            str.append("|");
            str.append(std::to_string(CurrentGapSize()));
            str.append("|");
        }
        if (m_pEnd - m_pGapEnd) {
            str.append((const char*)m_pGapEnd, (size_type)(m_pEnd - m_pGapEnd));
        }
        return str;
    }

    // Assign the whole buffer to this range of values
    template <class iter>
    void assign(iter srcBegin, iter srcEnd) {
        auto spaceRequired = std::distance(srcBegin, srcEnd);
        if (spaceRequired < 0)
            spaceRequired = -spaceRequired;

        // Make enough entries in the buffer
        resize((size_t)spaceRequired);

        // Copy directly to our buffer, using raw pointer for speed
        std::copy(srcBegin, srcEnd, m_pStart);

        // Gap is what's left at the end
        m_pGapStart = m_pStart + spaceRequired;
        m_pGapEnd = m_pEnd;

        assert(m_pGapStart <= m_pEnd);
        assert(size() >= size_t(spaceRequired));

        DEBUG_FILL_GAP;
    }

    void assign(std::initializer_list<T> list) {
        assign(list.begin(), list.end());
    }

    void assign(size_type count, const T& value) {
        // Make enough entries in the buffer
        resize((size_t)count);

        // Copy to the front of the buffer, leave the gap at that back
        T* p = m_pStart;
        while (p != m_pEnd) {
            *p = value;
            p++;
        }

        // Gap is what's left at the end
        m_pGapStart = m_pStart + count;
        m_pGapEnd = m_pEnd;

        assert(m_pGapStart <= m_pEnd);
        assert(size() >= size_t(count));

        DEBUG_FILL_GAP;
    }

    template <class iter>
    iterator insert(const_iterator pt, iter srcStart, iter srcEnd) {
        // Insert backwards - test it?
        auto spaceRequired = std::distance(srcStart, srcEnd);
        if (spaceRequired < 0)
            spaceRequired = -spaceRequired;

        EnsureGapPosAndSize(pt.p, spaceRequired);

        // Copy the data into the gap, move the start of gap pointer
        // Perf note: We pass a raw* to the buffer, because we want to copy with memmove.
        // We also don't pass an iterator because it would be used for incrementing pointers and negate the
        // useful mem copy
        std::copy(srcStart, srcEnd, m_pStart + pt.p);

        m_pGapStart += spaceRequired;

        DEBUG_FILL_GAP;

        return iterator(*this, pt.p);
    }

    iterator erase(const_iterator start, const_iterator end) {
        assert(start.p < size() && end.p < (size() + 1));
        MoveGap(start.p);

        auto count = std::distance(start, end);
        if (count < 0)
            count = -count;
        m_pGapEnd += count;

        DEBUG_FILL_GAP;

        return iterator(*this, start.p);
    }

    iterator erase(const_iterator start) {
        assert(start.p < size());
        MoveGap(start.p);
        m_pGapEnd++;

        DEBUG_FILL_GAP;

        return iterator(*this, start.p);
    }

    reference front() {
        assert(!empty());
        return *GetGaplessPtr(0);
    }

    const_reference front() const {
        assert(!empty());
        return *GetGaplessPtr(0);
    }

    reference back() {
        assert(!empty());
        return *GetGaplessPtr(size() - 1);
    }

    const_reference back() const {
        assert(!empty());
        return *GetGaplessPtr(size() - 1);
    }

    void push_front(const T& v) {
        EnsureGapPosAndSize(0, 1);

        *m_pGapStart = v;
        m_pGapStart++;
    }

    void push_front(T&& v) {
        EnsureGapPosAndSize(0, 1);

        *m_pGapStart = std::move(v);
        m_pGapStart++;
    }

    void push_back(const T& v) {
        EnsureGapPosAndSize(end().p, 1);
        *m_pGapStart = v;
        m_pGapStart++;
    }

    void push_back(T&& v) {
        EnsureGapPosAndSize(end().p, 1);
        *m_pGapStart = std::move(v);
        m_pGapStart++;
    }

    void pop_front() {
        assert(!empty());
        MoveGap(0);
        m_pGapEnd++;
    }

    void pop_back() {
        assert(!empty());
        MoveGap(end().p);
        m_pGapStart--;
    }

    reference operator[](size_type pos) {
        assert(pos < size());
        return *GetGaplessPtr(pos);
    }

    const_reference operator[](size_type pos) const {
        assert(pos < size());
        return *GetGaplessPtr(pos);
    }

    reference at(size_type pos) {
        assert(pos < size());
        return *GetGaplessPtr(pos);
    }
    const_reference at(size_type pos) const {
        assert(pos < size());
        return *GetGaplessPtr(pos);
    }

    // Here we split the find into 2 seperate searches; because we can be smart and search
    // either side of the gap.  This is more efficient that using an iterator which will keep
    // checking for the gap and trying to jump it
    template <class ForwardIt>
    T* find_first_of(T* pStart, T* pEnd, ForwardIt s_first, ForwardIt s_last) const {
        assert(pEnd <= m_pEnd);
        assert(pStart <= pEnd);
        while (pStart < m_pGapStart &&
               pStart < pEnd) {
            for (ForwardIt it = s_first; it != s_last; ++it) {
                if (*pStart == *it) {
                    assert(pStart <= pEnd);
                    return pStart;
                }
            }
            pStart++;
        }

        // Skip the gap
        if (pStart >= m_pGapStart && pStart < m_pGapEnd) {
            pStart = m_pGapEnd;
        }

        while (pStart < pEnd) {
            for (ForwardIt it = s_first; it != s_last; ++it) {
                if (*pStart == *it) {
                    assert(pStart <= pEnd);
                    return pStart;
                }
            }
            pStart++;
        }
        assert(pStart <= pEnd);
        return pEnd;
    }

    template <class ForwardIt>
    T* find_first_not_of(T* pStart, T* pEnd, ForwardIt s_first, ForwardIt s_last) const {
        bool found;
        while (pStart < m_pGapStart &&
               pStart < pEnd) {
            found = false;
            for (ForwardIt it = s_first; it != s_last; ++it) {
                if (*pStart == *it) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return pStart;
            }
            pStart++;
        }

        // Skip the gap
        if (pStart >= m_pGapStart && pStart < m_pGapEnd) {
            pStart = m_pGapEnd;
        }

        while (pStart < pEnd) {
            found = false;
            for (ForwardIt it = s_first; it != s_last; ++it) {
                if (*pStart == *it) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return pStart;
            }
            pStart++;
        }
        return pEnd;
    }

    // Wrappers around find_*
    template <class ForwardIt>
    const_iterator find_first_of(const_iterator first, const_iterator last,
                                 ForwardIt s_first, ForwardIt s_last) const {
        assert(first <= last);
        T* pVal = find_first_of(GetGaplessPtr(first.p), GetGaplessPtr(last.p), s_first, s_last);
        assert(GetGaplessPtr(first.p) <= pVal);
        auto itr = const_iterator(*this, GetGaplessOffset(pVal));
        // Return invalid if we walked to end without finding
        if (itr == last)
            itr = end();
        return itr;
    }

    template <class ForwardIt>
    const_iterator find_first_not_of(const_iterator first, const_iterator last,
                                     ForwardIt s_first, ForwardIt s_last) const {
        T* pVal = find_first_not_of(GetGaplessPtr(first.p), GetGaplessPtr(last.p), s_first, s_last);
        auto itr = const_iterator(*this, GetGaplessOffset(pVal));
        // Return invalid if we walked to end without finding
        if (itr == last)
            itr = end();
        return itr;
    }

    template <class ForwardIt>
    iterator find_first_of(iterator first, iterator last,
                           ForwardIt s_first, ForwardIt s_last) {
        assert(first <= last);
        T* pVal = find_first_of(GetGaplessPtr(first.p), GetGaplessPtr(last.p), s_first, s_last);
        assert(GetGaplessPtr(first.p) <= pVal);
        auto itr = iterator(*this, GetGaplessOffset(pVal));
        // Return invalid if we walked to end without finding
        if (itr == last)
            itr = end();
        return itr;
    }

    template <class ForwardIt>
    iterator find_first_not_of(iterator first, iterator last,
                               ForwardIt s_first, ForwardIt s_last) {
        T* pVal = find_first_not_of(GetGaplessPtr(first.p), GetGaplessPtr(last.p), s_first, s_last);
        auto itr = iterator(*this, GetGaplessOffset(pVal));
        // Return invalid if we walked to end without finding
        if (itr == last)
            itr = end();
        return itr;
    }

private:
    // Free everthing in the buffer
    // Note that clear() is like free, but also keeps an allocated Gap Buffer
    void Free() {
        if (m_pStart) {
            // Free all memory, including the gap
            get_allocator().deallocate(m_pStart, m_pEnd - m_pStart);
        }
        m_pStart = nullptr;
        m_pEnd = nullptr;
        m_pGapEnd = nullptr;
        m_pGapStart = nullptr;
    }

    // Return a buffer pos, but skip the gap - used by iterators to walk the buffer
    inline T* GetBufferPtr(size_t offset, bool skipGap = true) const {
        T* pos = m_pStart + offset;
        if (skipGap && pos >= m_pGapStart) {
            pos += m_pGapEnd - m_pGapStart;
        }

        return pos;
    }

    // TODO: Combine these if necessary for perf
    // We can probably be smarter about moving the gap if we are also resizing
    inline void EnsureGapPosAndSize(size_t p, size_t size) {
        MoveGap(p);

        if ((m_pGapEnd - m_pGapStart) < int64_t(size)) {
            resizeGap(size + m_defaultGap);
        }
    }

    // Get a pointer to the data ignoring the gap
    inline T* GetGaplessPtr(size_t offset) const {
        T* pos = m_pStart + offset;
        if (pos >= m_pGapStart) {
            pos += m_pGapEnd - m_pGapStart;
        }
        return pos;
    }

    // Get an offset to the data, ignoring the gap (external offset)
    inline size_t GetGaplessOffset(T* p) const {
        size_t offset = p - m_pStart;
        if (p >= m_pGapStart) {
            offset -= (m_pGapEnd - m_pGapStart);
        }
        return offset;
    }

    // The fixed_size of the buffer, including the gap
    inline size_t CurrentSizeWithGap() const {
        return m_pEnd - m_pStart;
    }

    // The fixed_size of the gap (will grow and shrink as necessary)
    inline size_t CurrentGapSize() const {
        return m_pGapEnd - m_pGapStart;
    }

    // Move the gap to before the given buffer position
    // As in most functions here, the pos is the offset in the buffer as if the 'gap' were invisible.
    // From the 'outside' this is just a linear vector of values.
    // From the 'inside', the gap is used to insert/remove items
    void MoveGap(size_t pos) {
        // Get a pointer, skip the gap
        T* pPos = m_pStart + pos;
        if (pPos >= m_pGapStart) {
            pPos += (m_pGapEnd - m_pGapStart);
        }

        // We are in the right place
        if (pPos == m_pGapEnd) {
            DEBUG_FILL_GAP;
            return;
        }

        // Move gap towards the left
        if (pPos < m_pGapStart) {
            // Move the gap start over by gapsize.
            memmove(pPos + (m_pGapEnd - m_pGapStart), pPos, m_pGapStart - pPos);
            m_pGapEnd -= (m_pGapStart - pPos);
            m_pGapStart = pPos;
        } else {
            // Since we are moving after the gap, find distance
            // between m_pGapEnd and target and that's how
            // much we move from m_pGapEnd to m_pGapStart.
            memmove(m_pGapStart, m_pGapEnd, pPos - m_pGapEnd);
            m_pGapStart += pPos - m_pGapEnd;
            m_pGapEnd = pPos;
        }
        DEBUG_FILL_GAP;
    }
};

/*

These functions are often available in containers - I just haven't implemented them yet ;)
Swap would be useful because it can be used to reset the memory usage if the gap has grown large
void swap(GapBuffer<T>&);

// Buffer comparisons
bool operator<(const GapBuffer<T>&) const;
bool operator>(const GapBuffer<T>&) const;
bool operator<=(const GapBuffer<T>&) const;
bool operator>=(const GapBuffer<T>&) const;

// Reverse iterators
reverse_iterator rbegin();
const_reverse_iterator rbegin() const;
const_reverse_iterator crbegin() const;
reverse_iterator rend();
const_reverse_iterator rend() const;
const_reverse_iterator crend() const;

// move support
template<class ...Args> void emplace_front(Args&&...);
template<class ...Args> void emplace_back(Args&&...);
template<class ...Args> iterator emplace(const_iterator, Args&&...);

// Different insertion types
iterator insert(const_iterator, const T&);
iterator insert(const_iterator, T&&);
iterator insert(const_iterator, size_type, T&);
iterator insert(const_iterator, std::initializer_list<T>);
*/
