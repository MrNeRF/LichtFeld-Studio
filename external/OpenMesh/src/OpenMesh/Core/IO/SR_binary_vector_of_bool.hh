
template <> struct binary< std::vector<bool> >
{
  typedef std::vector< bool >    value_type;
  typedef value_type::value_type elem_type;

  static const bool is_streamable = true;

  static size_t size_of(bool /*_store_size*/ = true) { return UnknownSize; }
  static size_t size_of(const value_type& _v, bool _store_size = true)
  {
    size_t size = _v.size() / 8 + ((_v.size() % 8)!=0);
    if(_store_size)
      size += binary<unsigned int>::size_of();
    return size;
  }
  static std::string type_identifier(void) { return "std::vector<bool>"; }
  static
  size_t store( std::ostream& _ostr, const value_type& _v, bool _swap, bool _store_size = true)
  {
    size_t bytes = 0;
    
    size_t N = _v.size() / 8;
    size_t R = _v.size() % 8;

    if(_store_size)
    {
      unsigned int size_N = static_cast<unsigned int>(_v.size());
      bytes += binary<unsigned int>::store( _ostr, size_N, _swap );
    }

    size_t        idx;  // element index
    size_t        bidx;
    unsigned char bits; // bitset

    for (bidx=idx=0; idx < N; ++idx, bidx+=8)
    {
      bits = static_cast<unsigned char>(_v[bidx])
        | (static_cast<unsigned char>(_v[bidx+1]) << 1)
        | (static_cast<unsigned char>(_v[bidx+2]) << 2)
        | (static_cast<unsigned char>(_v[bidx+3]) << 3)
        | (static_cast<unsigned char>(_v[bidx+4]) << 4)
        | (static_cast<unsigned char>(_v[bidx+5]) << 5)
        | (static_cast<unsigned char>(_v[bidx+6]) << 6)
        | (static_cast<unsigned char>(_v[bidx+7]) << 7);
      _ostr << bits;
    }
    bytes += N;

    if (R)
    {
      bits = 0;
      for (idx=0; idx < R; ++idx)
        bits |= static_cast<unsigned char>(_v[bidx+idx]) << idx;
      _ostr << bits;
      ++bytes;
    }
    assert( bytes == size_of(_v, _store_size) );

    return bytes;
  }

  static
  size_t restore( std::istream& _istr, value_type& _v, bool _swap, bool _restore_size = true)
  {
    size_t bytes = 0;

    if(_restore_size)
    {
      unsigned int size_of_vec;
      bytes += binary<unsigned int>::restore(_istr, size_of_vec, _swap);
      _v.resize(size_of_vec);
    }
    
    size_t N = _v.size() / 8;
    size_t R = _v.size() % 8;

    size_t        idx;  // element index
    size_t        bidx; //
    unsigned char bits; // bitset

    for (bidx=idx=0; idx < N; ++idx, bidx+=8)
    {
      _istr >> bits;
      _v[bidx+0] = (bits & 0x01) != 0;
      _v[bidx+1] = (bits & 0x02) != 0;
      _v[bidx+2] = (bits & 0x04) != 0;
      _v[bidx+3] = (bits & 0x08) != 0;
      _v[bidx+4] = (bits & 0x10) != 0;
      _v[bidx+5] = (bits & 0x20) != 0;
      _v[bidx+6] = (bits & 0x40) != 0;
      _v[bidx+7] = (bits & 0x80) != 0;
    }
    bytes += N;

    if (R)
    {
      _istr >> bits;
      for (idx=0; idx < R; ++idx)
        _v[bidx+idx] = (bits & (1<<idx)) != 0;
      ++bytes;
    }

    return bytes;
  }
};
