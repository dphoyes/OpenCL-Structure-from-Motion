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

    using namespace simd;

    // convolve image with a (1,4,6,4,1) row vector. Result is accumulated into output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_14641_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      auto i0 = (const array_8xint16_t*)(in);
      auto i1 = (const array_8xint16_t*)(in+1);
      auto i2 = (const array_8xint16_t*)(in+2);
      auto i3 = (const array_8xint16_t*)(in+3);
      auto i4 = (const array_8xint16_t*)(in+4);
      const auto end_input = (const array_8xint16_t*)(in + w*h);
      const auto offs = filled_array8<array_8xint16_t>(128);
      for(uint8_t* result = out+2; i4 < end_input; result += 16 )
      {
        array_8xint16_t result_registers[2];
        for(int i=0; i<2; i++, i0++, i1++, i2++, i3++, i4++)
        {
          const auto i1x4  = lshift_array_vals<2>(load_array(i1));
          const auto i0_4i1  = add_array_vals(load_array(i0), i1x4);
          const auto i2x2  = lshift_array_vals<1>(load_array(i2));
          const auto i2x4  = lshift_array_vals<2>(load_array(i2));
          const auto i2x6  = add_array_vals(i2x2, i2x4);
          const auto i0_4i1_6i2 = add_array_vals(i0_4i1, i2x6);
          const auto i3x4 = lshift_array_vals<2>(load_array(i3));
          const auto i0_4i1_6i2_4i3 = add_array_vals(i0_4i1_6i2, i3x4);
          const auto i0_4i1_6i2_4i3_i4 = add_array_vals(i0_4i1_6i2_4i3, load_array(i4));
          const auto scaled = rshift_array_vals<7>(i0_4i1_6i2_4i3_i4);
          result_registers[i] = add_array_vals(scaled, offs);
        }
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        store_array((array_16xuint8_t*)result, packed_result);
      }
    }

    // convolve image with a (1,2,0,-2,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_12021_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      auto i0 = (const array_8xint16_t*)(in);
      auto i1 = (const array_8xint16_t*)(in+1);
      auto i3 = (const array_8xint16_t*)(in+3);
      auto i4 = (const array_8xint16_t*)(in+4);
      const auto end_input = (const array_8xint16_t*)(in + w*h);
      const auto offs = filled_array8<array_8xint16_t>(128);
      for(uint8_t* result = out+2; i4 < end_input; result += 16 ) {
        array_8xint16_t result_registers[2];
        for( int i=0; i<2; i++, i0++, i1++, i3++, i4++) {
          const auto r1 = lshift_array_vals<1>(load_array(i1));
          const auto r2 = add_array_vals(r1, load_array(i0));
          const auto r3 = lshift_array_vals<1>(load_array(i3));
          const auto r4 = sub_array_vals(r2, r3);
          const auto r5 = sub_array_vals(r4, load_array(i4));
          const auto r6 = rshift_array_vals<7>(r5);
          result_registers[i] = add_array_vals(r6, offs);
        }
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        store_array((array_16xuint8_t*)result, packed_result);
      }
    }

    // convolve image with a (1,2,1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_121_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      auto i0 = (const array_8xint16_t*)(in);
      auto i1 = (const array_8xint16_t*)(in+1);
      auto i2 = (const array_8xint16_t*)(in+2);
      const auto offs = filled_array8<array_8xint16_t>(128);
      uint8_t* result   = out + 1;
      const size_t blocked_loops = (w*h-2)/16;
      for( size_t i=0; i != blocked_loops; i++, result += 16) {
        array_8xint16_t result_registers[2];
        for (unsigned lh=0; lh<2; lh++, i0++, i1++, i2++)
        {          
          const auto r1 = lshift_array_vals<1>(load_array(i1));
          const auto r2 = add_array_vals(r1, load_array(i0));
          const auto r3 = add_array_vals(load_array(i2), r2);
          const auto r4 = rshift_array_vals<2>(r3);
          result_registers[lh] = add_array_vals(r4, offs);
        }
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        store_array((array_16xuint8_t*)result, packed_result);
      }
    }
    
    // convolve image with a (1,0,-1) row vector. Result is accumulated into output.
    // This one works on 16bit input and 8bit output.
    // output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
    void convolve_101_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      auto i0 = (const array_8xint16_t*)(in);
      auto i2 = (const array_8xint16_t*)(in+2);
      const auto offs = filled_array8<array_8xint16_t>(128);
      uint8_t* result    = out + 1;
      const int16_t* const end_input = in + w*h;
      const size_t blocked_loops = (w*h-2)/16;
      for( size_t i=0; i != blocked_loops; i++, result += 16) {
        array_8xint16_t result_registers[2];
        for(unsigned lh=0; lh<2; lh++, i0++, i2++)
        {
          const auto r1 = sub_array_vals(load_array(i0), load_array(i2));
          const auto r2 = rshift_array_vals<2>(r1);
          result_registers[lh] = add_array_vals(r2, offs);
        }        
        const auto packed_result = array_pack_16to8(result_registers[0], result_registers[1]);
        store_array((array_16xuint8_t*)result, packed_result);
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
      auto i0              = (const array_16xuint8_t*)( in );
      auto i1              = (const array_16xuint8_t*)( in ) + w_chunk*1;
      auto i2              = (const array_16xuint8_t*)( in ) + w_chunk*2;
      auto i3              = (const array_16xuint8_t*)( in ) + w_chunk*3;
      auto i4              = (const array_16xuint8_t*)( in ) + w_chunk*4;
      const auto end_input = (const array_16xuint8_t*)( in ) + w_chunk*h;
      auto result_h        = (      array_8xint16_t*)( out_h ) + w_chunk*4;
      auto result_v        = (      array_8xint16_t*)( out_v ) + w_chunk*4;
      const auto sixes = filled_array8<array_8xint16_t>(6);
      const auto fours = filled_array8<array_8xint16_t>(4);
      for( ; i4 != end_input; i0++, i1++, i2++, i3++, i4++, result_v+=2, result_h+=2 ) {   
        const auto result_h1 = result_h+1;
        const auto result_v1 = result_v+1;
        const auto u0 = array_unpack_8to16(*i0);
        add_array_vals_in_place(*result_h, u0.hi);
        add_array_vals_in_place(*result_h1, u0.lo);
        add_array_vals_in_place(*result_v, u0.hi);
        add_array_vals_in_place(*result_v1, u0.lo);
        const auto u1 = array_unpack_8to16(*i1);
        add_array_vals_in_place(*result_h, u1.hi);
        add_array_vals_in_place(*result_h, u1.hi);
        add_array_vals_in_place(*result_h1, u1.lo);
        add_array_vals_in_place(*result_h1, u1.lo);
        const auto u1_x4 = mul_array_vals(u1, fours);
        add_array_vals_in_place(*result_v, u1_x4.hi);
        add_array_vals_in_place(*result_v1, u1_x4.lo);
        const auto u2 = array_unpack_8to16(*i2);
        const auto u2_x6 = mul_array_vals(u2, sixes);
        add_array_vals_in_place(*result_v, u2_x6.hi);
        add_array_vals_in_place(*result_v1, u2_x6.lo);
        const auto u3 = array_unpack_8to16(*i3);
        sub_array_vals_in_place(*result_h, u3.hi);
        sub_array_vals_in_place(*result_h, u3.hi);
        sub_array_vals_in_place(*result_h1, u3.lo);
        sub_array_vals_in_place(*result_h1, u3.lo);
        const auto u3_x4 = mul_array_vals(u3, fours);
        add_array_vals_in_place(*result_v, u3_x4.hi);
        add_array_vals_in_place(*result_v1, u3_x4.lo);
        const auto u4 = array_unpack_8to16(*i4);
        sub_array_vals_in_place(*result_h, u4.hi);
        sub_array_vals_in_place(*result_h1, u4.lo);
        add_array_vals_in_place(*result_v, u4.hi);
        add_array_vals_in_place(*result_v1, u4.lo);
      }
    }
    
    void convolve_col_p1p1p0m1m1_5x5( const unsigned char* in, int16_t* out, int w, int h ) {
      memset( out, 0, w*h*sizeof(int16_t) );
      using namespace std;
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      const int w_chunk  = w/16;
      auto i0              = (const array_16xuint8_t*)( in );
      auto i1              = (const array_16xuint8_t*)( in ) + w_chunk*1;
      auto i3              = (const array_16xuint8_t*)( in ) + w_chunk*3;
      auto i4              = (const array_16xuint8_t*)( in ) + w_chunk*4;
      const auto end_input = (const array_16xuint8_t*)( in ) + w_chunk*h;
      auto result          = (      array_8xint16_t*)( out ) + 4*w_chunk;
      for( ; i4 != end_input; i0++, i1++, i3++, i4++, result+=2 ) {
        const auto u0 = array_unpack_8to16(*i0);
        const auto u1 = array_unpack_8to16(*i1);
        *result     = add_array_vals(u0.hi, u1.hi);
        *(result+1) = add_array_vals(u0.lo, u1.lo);
        const auto u3 = array_unpack_8to16(*i3);
        sub_array_vals_in_place(*result, u3.hi);
        sub_array_vals_in_place(*(result+1), u3.lo);
        const auto u4 = array_unpack_8to16(*i4);
        sub_array_vals_in_place(*result, u4.hi);
        sub_array_vals_in_place(*(result+1), u4.lo);
      }
    }
    
    void convolve_row_p1p1p0m1m1_5x5( const int16_t* in, int16_t* out, int w, int h ) {
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      auto i0 = (const array_8xint16_t*)(in);
      auto i1 = (const array_8xint16_t*)(in+1);
      auto i3 = (const array_8xint16_t*)(in+3);
      auto i4 = (const array_8xint16_t*)(in+4);
      const auto end_input = (const array_8xint16_t*)(in + w*h);
      int16_t* result    = out + 2;
      for( ; i4+8 < end_input; i0++, i1++, i3++, i4++, result += 8 ) {
        const auto r0 = add_array_vals(load_array(i0), load_array(i1));
        const auto r1 = sub_array_vals(r0, load_array(i3));
        const auto r2 = sub_array_vals(r1, load_array(i4));
        store_array((array_8xint16_t*)result, r2);
      }
    }
    
    void convolve_cols_3x3( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
      using namespace std;
      assert( w % 16 == 0 && "width must be multiple of 16!" );
      const int w_chunk  = w/16;
      auto i0              = (const array_16xuint8_t*)( in );
      auto i1              = (const array_16xuint8_t*)( in ) + w_chunk*1;
      auto i2              = (const array_16xuint8_t*)( in ) + w_chunk*2;
      const auto end_input = (const array_16xuint8_t*)( in ) + w_chunk*h;
      auto result_h        = (      array_8xint16_t*)( out_h ) + 2*w_chunk;
      auto result_v        = (      array_8xint16_t*)( out_v ) + 2*w_chunk;
      for( ; i2 != end_input; i0++, i1++, i2++, result_v+=2, result_h+=2 ) {
        const auto u0 = array_unpack_8to16(*i0);
        *result_h     = u0.hi;
        *(result_h+1) = u0.lo;
        *result_v     = u0.hi;
        *(result_v+1) = u0.lo;
        const auto u1 = array_unpack_8to16(*i1);
        add_array_vals_in_place(*result_v, u1.hi);
        add_array_vals_in_place(*(result_v+1), u1.lo);
        add_array_vals_in_place(*result_v, u1.hi);
        add_array_vals_in_place(*(result_v+1), u1.lo);
        const auto u2 = array_unpack_8to16(*i2);
        sub_array_vals_in_place(*result_h, u2.hi);
        sub_array_vals_in_place(*(result_h+1), u2.lo);
        add_array_vals_in_place(*result_v, u2.hi);
        add_array_vals_in_place(*(result_v+1), u2.lo);
      }
    }
  }
  
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
