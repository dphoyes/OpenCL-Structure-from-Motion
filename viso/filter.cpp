/*
Copyright 2012. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Julius Ziegler, Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include <stdio.h>
#include <string.h>
#include <cassert>

#include "filter.h"

// define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

// fast filters: implements 3x3 and 5x5 sobel filters and 
//               5x5 blob and corner filters based on SSE2/3 instructions
namespace filter {
  
  // private namespace, public user functions at the bottom of this file
  namespace detail {
    void integral_image( const uint8_t* in, int32_t* out, int w, int h ) {
      int32_t* out_top = out;
      const uint8_t* line_end = in + w;
      const uint8_t* in_end   = in + w*h;
      int32_t line_sum = 0;
      for( ; in != line_end; in++, out++ ) {
        line_sum += *in;
        *out = line_sum;
      }
      for( ; in != in_end; ) {
        int32_t line_sum = 0;
        const uint8_t* line_end = in + w;
        for( ; in != line_end; in++, out++, out_top++ ) {
          line_sum += *in;
          *out = *out_top + line_sum;
        }
      }
    }

#if defined(USE_SIMD)
    void unpack_8bit_to_16bit( const __m128i a, __m128i& b0, __m128i& b1 ) {
      __m128i zero = _mm_setzero_si128();
      b0 = _mm_unpacklo_epi8( a, zero );
      b1 = _mm_unpackhi_epi8( a, zero );
    }
    
    void pack_16bit_to_8bit_saturate( const __m128i a0, const __m128i a1, __m128i& b ) {
      b = _mm_packus_epi16( a0, a1 );
    }
#endif

    template<size_t N>
    std::array<int16_t, N> add_array_vals(const std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
    {
      std::array<int16_t, N> res;
      for (size_t i=0; i<N; i++)
      {
        res[i] = a[i] + b[i];
      }
      return res;
    }

    template<size_t N>
    std::array<int16_t, N> sub_array_vals(const std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
    {
      std::array<int16_t, N> res;
      for (size_t i=0; i<N; i++)
      {
        res[i] = a[i] - b[i];
      }
      return res;
    }

    template<size_t N>
    void add_array_vals_in_place(std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
    {
      for (size_t i=0; i<N; i++)
      {
        a[i] += b[i];
      }
    }

    template<size_t N>
    void sub_array_vals_in_place(std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
    {
      for (size_t i=0; i<N; i++)
      {
        a[i] -= b[i];
      }
    }

    template<size_t N>
    std::pair< std::array<int16_t,N>, std::array<int16_t,N> > mul_array_vals(const std::pair< std::array<int16_t,N>, std::array<int16_t,N> > &a, const int multiplier)
    {
      std::pair< std::array<int16_t,N>, std::array<int16_t,N> > res;
      for (size_t i=0; i<N; i++)
      {
        res.first[i] = multiplier * a.first[i];
      }
      for (size_t i=0; i<N; i++)
      {
        res.second[i] = multiplier * a.second[i];
      }
      return res;
    }

    template<size_t N>
    std::array<int16_t, N> rshift_array_vals(const std::array<int16_t, N> &a, const unsigned shift_n)
    {
      std::array<int16_t, N> res;
      for (size_t i=0; i<N; i++)
      {
        res[i] = a[i] >> shift_n;
      }
      return res;
    }

    template<size_t N>
    std::array<uint8_t, 2*N> array_pack_16to8(const std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
    {
      auto saturate = [](int16_t x) -> uint8_t {return std::min(uint16_t(std::numeric_limits<uint8_t>::max()), uint16_t(x));};
      std::array<uint8_t, 2*N> res;
      for (size_t i=0; i<N; i++)
      {
        res[i] = saturate(a[i]);
      }
      for (size_t i=0; i<N; i++)
      {
        res[N+i] = saturate(b[i]);
      }
      return res;
    }

    template<size_t N>
    std::pair< std::array<int16_t,N/2>, std::array<int16_t,N/2> > array_unpack_8to16(const std::array<uint8_t, N> &a)
    {
      std::pair< std::array<int16_t,N/2>, std::array<int16_t,N/2> > res;

      for (size_t i=0; i<N/2; i++)
      {
        res.first[i] = a[i];
      }
      for (size_t i=0; i<N/2; i++)
      {
        res.second[i] = a[N/2+i];
      }
      return res;
    }
    
    // convolve image with a (1,4,6,4,1) row vector. Result is accumulated into output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_14641_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
#if defined(USE_SIMD)
      auto i0 = (const __m128i*)(in);
      auto i1 = (const __m128i*)(in+1);
      auto i2 = (const __m128i*)(in+2);
      auto i3 = (const __m128i*)(in+3);
      auto i4 = (const __m128i*)(in+4);
      const auto end_input = (const __m128i*)(in + w*h);
      const __m128i offs = _mm_set1_epi16( 128 );
#else
      auto i0 = (const std::array<int16_t, 8>*)(in);
      auto i1 = (const std::array<int16_t, 8>*)(in+1);
      auto i2 = (const std::array<int16_t, 8>*)(in+2);
      auto i3 = (const std::array<int16_t, 8>*)(in+3);
      auto i4 = (const std::array<int16_t, 8>*)(in+4);
      const auto end_input = (const std::array<int16_t, 8>*)(in + w*h);
      const std::array<int16_t, 8> offs = {128, 128, 128, 128, 128, 128, 128, 128};
#endif
      uint8_t* result   = out + 2;
      for( ; i4 < end_input; result += 16 ) {
#if defined(USE_SIMD)
        __m128i result_register_lo, result_register_hi;
        for( int i=0; i<2; i++ ) {
          __m128i* result_register = i==0 ? &result_register_lo : &result_register_hi;
          __m128i i0_register = _mm_loadu_si128(i0);
          __m128i i1_register = _mm_loadu_si128(i1);
          __m128i i2_register = _mm_loadu_si128(i2);
          __m128i i3_register = _mm_loadu_si128(i3);
          __m128i i4_register = _mm_loadu_si128(i4);
          *result_register = _mm_setzero_si128();
          *result_register = _mm_add_epi16( i0_register, *result_register );
          i1_register      = _mm_add_epi16( i1_register, i1_register  );
          i1_register      = _mm_add_epi16( i1_register, i1_register  );
          *result_register = _mm_add_epi16( i1_register, *result_register );
          i2_register      = _mm_add_epi16( i2_register, i2_register  );
          *result_register = _mm_add_epi16( i2_register, *result_register );
          i2_register      = _mm_add_epi16( i2_register, i2_register  );
          *result_register = _mm_add_epi16( i2_register, *result_register );
          i3_register      = _mm_add_epi16( i3_register, i3_register  );
          i3_register      = _mm_add_epi16( i3_register, i3_register  );
          *result_register = _mm_add_epi16( i3_register, *result_register );
          *result_register = _mm_add_epi16( i4_register, *result_register );
          *result_register = _mm_srai_epi16( *result_register, 7 );
          *result_register = _mm_add_epi16( *result_register, offs );

          i0 += 1; i1 += 1; i2 += 1; i3 += 1; i4 += 1;
        }
        pack_16bit_to_8bit_saturate( result_register_lo, result_register_hi, result_register_lo );
        _mm_storeu_si128( ((__m128i*)( result )), result_register_lo );
#else
        std::array<int16_t, 8> result_register_lo, result_register_hi;
        for( int i=0; i<2; i++ )
        {
          const auto result_reg = i==0 ? &result_register_lo : &result_register_hi;

          const auto r1  = add_array_vals(*i1, *i1);
          const auto r2  = add_array_vals(r1, r1);
          const auto r3  = add_array_vals(r2, *i0);
          const auto r4  = add_array_vals(*i2, *i2);
          const auto r5  = add_array_vals(r4, r3);
          const auto r6  = add_array_vals(r4, r4);
          const auto r7  = add_array_vals(r6, r5);
          const auto r8  = add_array_vals(*i3, *i3);
          const auto r9 = add_array_vals(r8, r8);
          const auto r10 = add_array_vals(r9, r7);
          const auto r11 = add_array_vals(*i4, r10);
          const auto r12 = rshift_array_vals(r11, 7);
          *result_reg = add_array_vals(r12, offs);

          i0 += 1; i1 += 1; i2 += 1; i3 += 1; i4 += 1;
        }
        const auto packed_result = array_pack_16to8(result_register_lo, result_register_hi);
        *(std::array<uint8_t, 16>*)result = packed_result;
#endif
      }
    }
    
    // convolve image with a (1,2,0,-2,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_12021_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
#if defined(USE_SIMD)
      auto i0 = (const __m128i*)(in);
      auto i1 = (const __m128i*)(in+1);
      auto i3 = (const __m128i*)(in+3);
      auto i4 = (const __m128i*)(in+4);
      const auto end_input = (const __m128i*)(in + w*h);
      __m128i offs = _mm_set1_epi16( 128 );
#else
      auto i0 = (const std::array<int16_t, 8>*)(in);
      auto i1 = (const std::array<int16_t, 8>*)(in+1);
      auto i3 = (const std::array<int16_t, 8>*)(in+3);
      auto i4 = (const std::array<int16_t, 8>*)(in+4);
      const auto end_input = (const std::array<int16_t, 8>*)(in + w*h);
      const std::array<int16_t, 8> offs = {128, 128, 128, 128, 128, 128, 128, 128};
#endif
      uint8_t* result    = out + 2;
      for( ; i4 < end_input; result += 16 ) {
#if defined(USE_SIMD)
        __m128i result_register_lo, result_register_hi;
        for( int i=0; i<2; i++ ) {
          __m128i* result_register = i==0 ? &result_register_lo : &result_register_hi;
          __m128i i0_register = _mm_loadu_si128(i0);
          __m128i i1_register = _mm_loadu_si128(i1);
          __m128i i3_register = _mm_loadu_si128(i3);
          __m128i i4_register = _mm_loadu_si128(i4);
          *result_register = _mm_setzero_si128();
          *result_register = _mm_add_epi16( i0_register,   *result_register );
          i1_register      = _mm_add_epi16( i1_register, i1_register  );
          *result_register = _mm_add_epi16( i1_register,   *result_register );
          i3_register      = _mm_add_epi16( i3_register, i3_register  );
          *result_register = _mm_sub_epi16( *result_register, i3_register );
          *result_register = _mm_sub_epi16( *result_register, i4_register );
          *result_register = _mm_srai_epi16( *result_register, 7 );
          *result_register = _mm_add_epi16( *result_register, offs );

          i0 += 1; i1 += 1; i3 += 1; i4 += 1;
        }
        pack_16bit_to_8bit_saturate( result_register_lo, result_register_hi, result_register_lo );
        _mm_storeu_si128( ((__m128i*)( result )), result_register_lo );
#else
        std::array<int16_t, 8> result_registers[2];
        for( int i=0; i<2; i++, i0++, i1++, i3++, i4++) {
          const auto r1 = add_array_vals(*i1, *i1);
          const auto r2 = add_array_vals(r1, *i0);
          const auto r3 = add_array_vals(*i3, *i3);
          const auto r4 = sub_array_vals(r2, r3);
          const auto r5 = sub_array_vals(r4, *i4);
          const auto r6 = rshift_array_vals(r5, 7);
          result_registers[i] = add_array_vals(r6, offs);
        }
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        *(std::array<uint8_t, 16>*)result = packed_result;
#endif
      }
    }

    // convolve image with a (1,2,1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_121_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
#if defined(USE_SIMD)
      auto i0 = (const __m128i*)(in);
      auto i1 = (const __m128i*)(in+1);
      auto i2 = (const __m128i*)(in+2);
      __m128i offs = _mm_set1_epi16( 128 );
#else
      auto i0 = (const std::array<int16_t, 8>*)(in);
      auto i1 = (const std::array<int16_t, 8>*)(in+1);
      auto i2 = (const std::array<int16_t, 8>*)(in+2);
      const std::array<int16_t, 8> offs = {128, 128, 128, 128, 128, 128, 128, 128};
#endif
      uint8_t* result   = out + 1;
      const size_t blocked_loops = (w*h-2)/16;
      for( size_t i=0; i != blocked_loops; i++, result += 16) {
#if defined(USE_SIMD)
        __m128i result_registers[2];
        __m128i i1_register;
        __m128i i2_register;
        for (unsigned lh=0; lh<2; lh++, i0++, i1++, i2++)
        {
          __m128i* result_register = &result_registers[lh];
          i1_register        = _mm_loadu_si128(i1);
          i2_register        = _mm_loadu_si128(i2);
          *result_register = _mm_loadu_si128(i0);
          i1_register        = _mm_add_epi16( i1_register, i1_register );
          *result_register = _mm_add_epi16( i1_register, *result_register );
          *result_register = _mm_add_epi16( i2_register, *result_register );
          *result_register = _mm_srai_epi16( *result_register, 2 );
          *result_register = _mm_add_epi16( *result_register, offs );
        }
        pack_16bit_to_8bit_saturate( result_registers[0], result_registers[1], result_registers[0] );
        _mm_storeu_si128( ((__m128i*)( result )), result_registers[0] );
#else
        std::array<int16_t, 8> result_registers[2];
        for (unsigned lh=0; lh<2; lh++, i0++, i1++, i2++)
        {          
          const auto r1 = add_array_vals(*i1, *i1);
          const auto r2 = add_array_vals(r1, *i0);
          const auto r3 = add_array_vals(*i2, r2);
          const auto r4 = rshift_array_vals(r3, 2);
          result_registers[lh] = add_array_vals( r4, offs );
        }
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        *(std::array<uint8_t, 16>*)result = packed_result;
#endif        
      }
    }
    
    // convolve image with a (1,0,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_101_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
#if defined(USE_SIMD)
      auto i0 = (const __m128i*)(in);
      auto i2 = (const __m128i*)(in+2);
      const __m128i offs = _mm_set1_epi16( 128 );
#else
      auto i0 = (const std::array<int16_t, 8>*)(in);
      auto i2 = (const std::array<int16_t, 8>*)(in+2);
      const std::array<int16_t, 8> offs = {128, 128, 128, 128, 128, 128, 128, 128};
#endif
      uint8_t* result    = out + 1;
      const int16_t* const end_input = in + w*h;
      const size_t blocked_loops = (w*h-2)/16;
      for( size_t i=0; i != blocked_loops; i++, result += 16) {
#if defined(USE_SIMD)
        __m128i result_registers[2];
        __m128i i2_register;
        for(unsigned lh=0; lh<2; lh++, i0++, i2++)
        {
          __m128i* result_register = &result_registers[lh];
          i2_register = _mm_loadu_si128(i2);
          *result_register  = _mm_loadu_si128(i0);
          *result_register  = _mm_sub_epi16( *result_register, i2_register );
          *result_register  = _mm_srai_epi16( *result_register, 2 );
          *result_register  = _mm_add_epi16( *result_register, offs );
        }        
        pack_16bit_to_8bit_saturate( result_registers[0], result_registers[1], result_registers[0] );
        _mm_storeu_si128( ((__m128i*)( result )), result_registers[0] );
#else
        std::array<int16_t, 8> result_registers[2];
        for(unsigned lh=0; lh<2; lh++, i0++, i2++)
        {
          const auto r1 = sub_array_vals(*i0, *i2);
          const auto r2 = rshift_array_vals(r1, 2);
          result_registers[lh] = add_array_vals(r2, offs);
        }        
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        *(std::array<uint8_t, 16>*)result = packed_result;
#endif
      }

      for(auto _i2 = (int16_t*)i2; _i2 < end_input; _i2++, result++) {
        *result = ((*(_i2-2) - *_i2)>>2)+128;
      }
    }
    
    void convolve_cols_5x5( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
      using namespace std;
      memset( out_h, 0, w*h*sizeof(int16_t) );
      memset( out_v, 0, w*h*sizeof(int16_t) );
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      const int w_chunk  = w/16;
#if defined(USE_SIMD)
      auto i0        = (__m128i*)( in );
      auto i1        = (__m128i*)( in ) + w_chunk*1;
      auto i2        = (__m128i*)( in ) + w_chunk*2;
      auto i3        = (__m128i*)( in ) + w_chunk*3;
      auto i4        = (__m128i*)( in ) + w_chunk*4;
      auto result_h  = (__m128i*)( out_h ) + 4*w_chunk;
      auto result_v  = (__m128i*)( out_v ) + 4*w_chunk;
      auto end_input = (__m128i*)( in ) + w_chunk*h;
      __m128i sixes      = _mm_set1_epi16( 6 );
      __m128i fours      = _mm_set1_epi16( 4 );
#else
      auto i0        = (const std::array<uint8_t, 16>*)( in );
      auto i1        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*1;
      auto i2        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*2;
      auto i3        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*3;
      auto i4        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*4;
      auto result_h  = (      std::array<int16_t, 8>*)( out_h ) + 4*w_chunk;
      auto result_v  = (      std::array<int16_t, 8>*)( out_v ) + 4*w_chunk;
      auto end_input = (const std::array<uint8_t, 16>*)( in ) + w_chunk*h;
#endif
      for( ; i4 != end_input; i0++, i1++, i2++, i3++, i4++, result_v+=2, result_h+=2 ) {   
#if defined(USE_SIMD)
        __m128i ilo, ihi;
        unpack_8bit_to_16bit( *i0, ihi, ilo );
        *result_h     = _mm_add_epi16( ihi, *result_h );
        *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        unpack_8bit_to_16bit( *i1, ihi, ilo );
        *result_h     = _mm_add_epi16( ihi, *result_h );
        *result_h     = _mm_add_epi16( ihi, *result_h );
        *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
        *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
        ihi = _mm_mullo_epi16( ihi, fours );
        ilo = _mm_mullo_epi16( ilo, fours );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        unpack_8bit_to_16bit( *i2, ihi, ilo );
        ihi = _mm_mullo_epi16( ihi, sixes );
        ilo = _mm_mullo_epi16( ilo, sixes );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        unpack_8bit_to_16bit( *i3, ihi, ilo );
        *result_h     = _mm_sub_epi16( *result_h, ihi );
        *result_h     = _mm_sub_epi16( *result_h, ihi );
        *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
        *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
        ihi = _mm_mullo_epi16( ihi, fours );
        ilo = _mm_mullo_epi16( ilo, fours );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );          
        unpack_8bit_to_16bit( *i4, ihi, ilo );
        *result_h     = _mm_sub_epi16( *result_h, ihi );
        *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
#else
        const auto result_h1 = result_h+1;
        const auto result_v1 = result_v+1;
        const auto u0 = array_unpack_8to16(*i0);
        add_array_vals_in_place(*result_h, u0.first);
        add_array_vals_in_place(*result_h1, u0.second);
        add_array_vals_in_place(*result_v, u0.first);
        add_array_vals_in_place(*result_v1, u0.second);
        const auto u1 = array_unpack_8to16(*i1);
        add_array_vals_in_place(*result_h, u1.first);
        add_array_vals_in_place(*result_h, u1.first);
        add_array_vals_in_place(*result_h1, u1.second);
        add_array_vals_in_place(*result_h1, u1.second);
        const auto u1_x4 = mul_array_vals(u1, 4);
        add_array_vals_in_place(*result_v, u1_x4.first);
        add_array_vals_in_place(*result_v1, u1_x4.second);
        const auto u2 = array_unpack_8to16(*i2);
        const auto u2_x6 = mul_array_vals(u2, 6);
        add_array_vals_in_place(*result_v, u2_x6.first);
        add_array_vals_in_place(*result_v1, u2_x6.second);
        const auto u3 = array_unpack_8to16(*i3);
        sub_array_vals_in_place(*result_h, u3.first);
        sub_array_vals_in_place(*result_h, u3.first);
        sub_array_vals_in_place(*result_h1, u3.second);
        sub_array_vals_in_place(*result_h1, u3.second);
        const auto u3_x4 = mul_array_vals(u3, 4);
        add_array_vals_in_place(*result_v, u3_x4.first);
        add_array_vals_in_place(*result_v1, u3_x4.second);
        const auto u4 = array_unpack_8to16(*i4);
        sub_array_vals_in_place(*result_h, u4.first);
        sub_array_vals_in_place(*result_h1, u4.second);
        add_array_vals_in_place(*result_v, u4.first);
        add_array_vals_in_place(*result_v1, u4.second);
#endif
      }
    }
    
    // possible that non-sse is better here?
    void convolve_col_p1p1p0m1m1_5x5( const unsigned char* in, int16_t* out, int w, int h ) {
      memset( out, 0, w*h*sizeof(int16_t) );
      using namespace std;
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      const int w_chunk  = w/16;
#if defined(USE_SIMD)
      auto i0       = (__m128i*)( in );
      auto i1       = (__m128i*)( in ) + w_chunk*1;
      auto i3       = (__m128i*)( in ) + w_chunk*3;
      auto i4       = (__m128i*)( in ) + w_chunk*4;
      auto result   = (__m128i*)( out ) + 4*w_chunk;
      const auto end_input = (__m128i*)( in ) + w_chunk*h;
#else
      auto i0       = (const std::array<uint8_t, 16>*)( in );
      auto i1       = (const std::array<uint8_t, 16>*)( in ) + w_chunk*1;
      auto i3       = (const std::array<uint8_t, 16>*)( in ) + w_chunk*3;
      auto i4       = (const std::array<uint8_t, 16>*)( in ) + w_chunk*4;
      auto result   = (      std::array<int16_t, 8>*)( out ) + 4*w_chunk;
      const auto end_input = (const std::array<uint8_t, 16>*)( in ) + w_chunk*h;
#endif
      for( ; i4 != end_input; i0++, i1++, i3++, i4++, result+=2 ) {
#if defined(USE_SIMD)
        __m128i ilo0, ihi0;
        unpack_8bit_to_16bit( *i0, ihi0, ilo0 );
        __m128i ilo1, ihi1;
        unpack_8bit_to_16bit( *i1, ihi1, ilo1 );
        *result     = _mm_add_epi16( ihi0, ihi1 );
        *(result+1) = _mm_add_epi16( ilo0, ilo1 );
        __m128i ilo, ihi;
        unpack_8bit_to_16bit( *i3, ihi, ilo );
        *result     = _mm_sub_epi16( *result, ihi );
        *(result+1) = _mm_sub_epi16( *(result+1), ilo );
        unpack_8bit_to_16bit( *i4, ihi, ilo );
        *result     = _mm_sub_epi16( *result, ihi );
        *(result+1) = _mm_sub_epi16( *(result+1), ilo );
#else
        const auto u0 = array_unpack_8to16(*i0);
        const auto u1 = array_unpack_8to16(*i1);
        *result     = add_array_vals(u0.second, u1.second);
        *(result+1) = add_array_vals(u0.first, u1.first);
        const auto u3 = array_unpack_8to16(*i3);
        *result     = sub_array_vals(*result, u3.second);
        *(result+1) = sub_array_vals(*(result+1), u3.first);
        const auto u4 = array_unpack_8to16(*i4);
        *result     = sub_array_vals(*result, u4.second);
        *(result+1) = sub_array_vals(*(result+1), u4.first);
#endif
      }
    }
    
    void convolve_row_p1p1p0m1m1_5x5( const int16_t* in, int16_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
#if defined(USE_SIMD)
      auto i0 = (const __m128i*)(in);
      auto i1 = (const __m128i*)(in+1);
      auto i3 = (const __m128i*)(in+3);
      auto i4 = (const __m128i*)(in+4);
      const auto end_input = (const __m128i*)(in + w*h);
#else
      auto i0 = (const std::array<int16_t, 8>*)(in);
      auto i1 = (const std::array<int16_t, 8>*)(in+1);
      auto i3 = (const std::array<int16_t, 8>*)(in+3);
      auto i4 = (const std::array<int16_t, 8>*)(in+4);
      const auto end_input = (const std::array<int16_t, 8>*)(in + w*h);
#endif
      int16_t* result    = out + 2;
      for( ; i4+8 < end_input; i0++, i1++, i3++, i4++, result += 8 ) {
#if defined(USE_SIMD)
        __m128i result_register;
        __m128i i0_register = _mm_loadu_si128(i0);
        __m128i i1_register = _mm_loadu_si128(i1);
        __m128i i3_register = _mm_loadu_si128(i3);
        __m128i i4_register = _mm_loadu_si128(i4);
        result_register     = _mm_add_epi16( i0_register,     i1_register );
        result_register     = _mm_sub_epi16( result_register, i3_register );
        result_register     = _mm_sub_epi16( result_register, i4_register );
        _mm_storeu_si128( ((__m128i*)( result )), result_register );
#else
        const auto r0 = add_array_vals(*i0, *i1);
        const auto r1 = sub_array_vals(r0,  *i3);
        const auto r2 = sub_array_vals(r1,  *i4);
        *(std::array<int16_t, 8>*)result = r2;
#endif
      }
    }
    
    // possible that non-sse is better here?
    void convolve_cols_3x3( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
      using namespace std;
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      const int w_chunk  = w/16;
#if defined(USE_SIMD)
      auto i0        = (__m128i*)( in );
      auto i1        = (__m128i*)( in ) + w_chunk*1;
      auto i2        = (__m128i*)( in ) + w_chunk*2;
      auto result_h  = (__m128i*)( out_h ) + 2*w_chunk;
      auto result_v  = (__m128i*)( out_v ) + 2*w_chunk;
      const auto end_input = (__m128i*)( in ) + w_chunk*h;
#else
      auto i0        = (const std::array<uint8_t, 16>*)( in );
      auto i1        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*1;
      auto i2        = (const std::array<uint8_t, 16>*)( in ) + w_chunk*2;
      auto result_h  = (      std::array<int16_t, 8>*)( out_h ) + 2*w_chunk;
      auto result_v  = (      std::array<int16_t, 8>*)( out_v ) + 2*w_chunk;
      const auto end_input = (const std::array<uint8_t, 16>*)( in ) + w_chunk*h;
#endif
      for( ; i2 != end_input; i0++, i1++, i2++, result_v+=2, result_h+=2 ) {
#if defined(USE_SIMD)
        *result_h     = _mm_setzero_si128();
        *(result_h+1) = _mm_setzero_si128();
        *result_v     = _mm_setzero_si128();
        *(result_v+1) = _mm_setzero_si128();
        __m128i ilo, ihi;
        unpack_8bit_to_16bit( *i0, ihi, ilo );
        *result_h     = _mm_add_epi16( ihi, *result_h );
        *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        unpack_8bit_to_16bit( *i1, ihi, ilo );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
        unpack_8bit_to_16bit( *i2, ihi, ilo );
        *result_h     = _mm_sub_epi16( *result_h, ihi );
        *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
        *result_v     = _mm_add_epi16( *result_v, ihi );
        *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
#else
        const auto u0 = array_unpack_8to16(*i0);
        *result_h     = u0.second;
        *(result_h+1) = u0.first;
        *result_v     = u0.second;
        *(result_v+1) = u0.first;
        const auto u1 = array_unpack_8to16(*i1);
        *result_v     = add_array_vals(*result_v, u1.second);
        *(result_v+1) = add_array_vals(*(result_v+1), u1.first);
        *result_v     = add_array_vals(*result_v, u1.second);
        *(result_v+1) = add_array_vals(*(result_v+1), u1.first);
        const auto u2 = array_unpack_8to16(*i2);
        *result_h     = sub_array_vals(*result_h, u2.second);
        *(result_h+1) = sub_array_vals(*(result_h+1), u2.first);
        *result_v     = add_array_vals(*result_v, u2.second);
        *(result_v+1) = add_array_vals(*(result_v+1), u2.first);
#endif
      }
    }
  };
  
  void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h ) {
    int16_t* temp_h = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
    int16_t* temp_v = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );    
    detail::convolve_cols_3x3( in, temp_v, temp_h, w, h );
    detail::convolve_101_row_3x3_16bit( temp_v, out_v, w, h );
    detail::convolve_121_row_3x3_16bit( temp_h, out_h, w, h );
    _mm_free( temp_h );
    _mm_free( temp_v );
  }
  
  void sobel5x5( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h ) {
    int16_t* temp_h = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
    int16_t* temp_v = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
    detail::convolve_cols_5x5( in, temp_v, temp_h, w, h );
    detail::convolve_12021_row_5x5_16bit( temp_v, out_v, w, h );
    detail::convolve_14641_row_5x5_16bit( temp_h, out_h, w, h );
    _mm_free( temp_h );
    _mm_free( temp_v );
  }
  
  // -1 -1  0  1  1
  // -1 -1  0  1  1
  //  0  0  0  0  0
  //  1  1  0 -1 -1
  //  1  1  0 -1 -1
  void checkerboard5x5( const uint8_t* in, int16_t* out, int w, int h ) {
    int16_t* temp = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
    detail::convolve_col_p1p1p0m1m1_5x5( in, temp, w, h );
    detail::convolve_row_p1p1p0m1m1_5x5( temp, out, w, h );
    _mm_free( temp );
  }
  
  // -1 -1 -1 -1 -1
  // -1  1  1  1 -1
  // -1  1  8  1 -1
  // -1  1  1  1 -1
  // -1 -1 -1 -1 -1
  void blob5x5( const uint8_t* in, int16_t* out, int w, int h ) {
    int32_t* integral = (int32_t*)( _mm_malloc( w*h*sizeof( int32_t ), 16 ) );
    detail::integral_image( in, integral, w, h );
    int16_t* out_ptr   = out + 3 + 3*w;
    int16_t* out_end   = out + w * h - 2 - 2*w;
    const int32_t* i00 = integral;
    const int32_t* i50 = integral + 5;
    const int32_t* i05 = integral + 5*w;
    const int32_t* i55 = integral + 5 + 5*w;
    const int32_t* i11 = integral + 1 + 1*w;
    const int32_t* i41 = integral + 4 + 1*w;
    const int32_t* i14 = integral + 1 + 4*w;
    const int32_t* i44 = integral + 4 + 4*w;    
    const uint8_t* im22 = in + 3 + 3*w;
    for( ; out_ptr != out_end; out_ptr++, i00++, i50++, i05++, i55++, i11++, i41++, i14++, i44++, im22++ ) {
      int32_t result = 0;
      result = -( *i55 - *i50 - *i05 + *i00 );
      result += 2*( *i44 - *i41 - *i14 + *i11 );
      result += 7* *im22;
      *out_ptr = result;
    }
    _mm_free( integral );
  }
};
