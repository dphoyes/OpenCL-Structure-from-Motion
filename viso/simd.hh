#ifndef SIMD_HH
#define SIMD_HH

#include <array>
#include <limits>
#include <algorithm>

namespace simd
{
#ifdef __SSE3__
    #include <emmintrin.h>
    #include <pmmintrin.h>
    struct __m128i_int16x8 {__m128i val;};
    struct __m128i_uint8x16 {__m128i val;};
#endif
#ifdef __ARM_NEON__
    #include <arm_neon.h>
#endif

#if defined(USE_SIMD)
    #if defined(__SSE3__)
        typedef __m128i_int16x8 array_8xint16_t;
        typedef __m128i_uint8x16 array_16xuint8_t;
    #elif defined(__ARM_NEON__)
        typedef int16x8_t array_8xint16_t;
        typedef uint8x16_t array_16xuint8_t;
    #else
        #error No SIMD implementation available
    #endif
#else
    typedef std::array<int16_t, 8> array_8xint16_t;
    typedef std::array<uint8_t, 16> array_16xuint8_t;
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
    #ifdef __SSE3__
    inline __m128i_int16x8 __attribute__((__always_inline__)) add_array_vals(const __m128i_int16x8 &a, const __m128i_int16x8 &b)
    {
        return __m128i_int16x8{_mm_add_epi16(a.val, b.val)};
    }
    #endif
    #ifdef __ARM_NEON__
    inline int16x8_t __attribute__((__always_inline__)) add_array_vals(const int16x8_t &a, const int16x8_t &b)
    {
        return vaddq_s16(a,b);
    }
    #endif

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
    #ifdef __SSE3__
    inline __m128i_int16x8 __attribute__((__always_inline__)) sub_array_vals(const __m128i_int16x8 &a, const __m128i_int16x8 &b)
    {
        return __m128i_int16x8{_mm_sub_epi16(a.val, b.val)};
    }
    #endif
    #ifdef __ARM_NEON__
    inline int16x8_t __attribute__((__always_inline__)) sub_array_vals(const int16x8_t &a, const int16x8_t &b)
    {
            return vsubq_s16(a,b);
    }
    #endif

template<size_t N>
void add_array_vals_in_place(std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
{
    for (size_t i=0; i<N; i++)
    {
        a[i] += b[i];
    }
}
    #ifdef __SSE3__
    inline void __attribute__((__always_inline__)) add_array_vals_in_place(__m128i_int16x8 &a, const __m128i_int16x8 &b)
    {
        a.val = _mm_add_epi16(a.val, b.val);
    }
    #endif
    #ifdef __ARM_NEON__
    inline void __attribute__((__always_inline__)) add_array_vals_in_place(int16x8_t &a, const int16x8_t &b)
    {
        a = vaddq_s16(a,b);
    }
    #endif

template<size_t N>
void sub_array_vals_in_place(std::array<int16_t, N> &a, const std::array<int16_t, N> &b)
{
    for (size_t i=0; i<N; i++)
    {
        a[i] -= b[i];
    }
}
    #ifdef __SSE3__
    inline void __attribute__((__always_inline__)) sub_array_vals_in_place(__m128i_int16x8 &a, const __m128i_int16x8 &b)
    {
        a.val = _mm_sub_epi16(a.val, b.val);
    }
    #endif
    #ifdef __ARM_NEON__
    inline void __attribute__((__always_inline__)) sub_array_vals_in_place(int16x8_t &a, const int16x8_t &b)
    {
        a = vsubq_s16(a,b);
    }
    #endif

template<size_t N>
struct unpacked16arrays_t
{
    std::array<int16_t,N> hi;
    std::array<int16_t,N> lo;
};
#ifdef __SSE3__
struct __m128i_pair
{
    __m128i_int16x8 hi;
    __m128i_int16x8 lo;
};
#endif
#ifdef __ARM_NEON__
struct int16x8_pair_t
{
    int16x8_t hi;
    int16x8_t lo;
};
#endif

template<size_t N>
unpacked16arrays_t<N> mul_array_vals(const unpacked16arrays_t<N> &a, const std::array<int16_t,N> &multiplier)
{
    unpacked16arrays_t<N> res;
    for (size_t i=0; i<N; i++)
    {
        res.hi[i] = multiplier[i] * a.hi[i];
    }
    for (size_t i=0; i<N; i++)
    {
        res.lo[i] = multiplier[i] * a.lo[i];
    }
    return res;
}
    #ifdef __SSE3__
    inline __m128i_pair __attribute__((__always_inline__)) mul_array_vals(const __m128i_pair &a, const __m128i_int16x8 &b)
    {
        __m128i_pair res;
        res.hi.val = _mm_mullo_epi16(a.hi.val, b.val);
        res.lo.val = _mm_mullo_epi16(a.lo.val, b.val);
        return res;
    }
    #endif
    #ifdef __ARM_NEON__
    inline int16x8_pair_t __attribute__((__always_inline__)) mul_array_vals(const int16x8_pair_t &a, const int16x8_t &b)
    {
        int16x8_pair_t res;
        res.hi = vmulq_s16(a.hi, b);
        res.lo = vmulq_s16(a.lo, b);
        return res;
    }
    #endif

template<int shift_n>
std::array<int16_t, 8> rshift_array_vals(const std::array<int16_t, 8> &a)
{
    static const unsigned N = 8;
    std::array<int16_t, N> res;
    for (size_t i=0; i<N; i++)
    {
        res[i] = a[i] >> shift_n;
    }
    return res;
}
    #ifdef __SSE3__
    template<int shift_n>
    inline __m128i_int16x8 __attribute__((__always_inline__)) rshift_array_vals(const __m128i_int16x8 &a)
    {
        return __m128i_int16x8{_mm_srai_epi16(a.val, shift_n)};
    }
    #endif
    #ifdef __ARM_NEON__
    template<int shift_n>
    inline int16x8_t __attribute__((__always_inline__)) rshift_array_vals(const int16x8_t &a)
    {
        return vshrq_n_s16(a, shift_n);
    }
    #endif

template<int shift_n>
std::array<int16_t, 8> lshift_array_vals(const std::array<int16_t, 8> &a)
{
    static const unsigned N = 8;
    std::array<int16_t, N> res;
    for (size_t i=0; i<N; i++)
    {
        res[i] = a[i] << shift_n;
    }
    return res;
}
    #ifdef __SSE3__
    template<int shift_n>
    inline __m128i_int16x8 __attribute__((__always_inline__)) lshift_array_vals(const __m128i_int16x8 &a)
    {
        return __m128i_int16x8{_mm_slli_epi16(a.val, shift_n)};
    }
    #endif
    #ifdef __ARM_NEON__
    template<int shift_n>
    inline int16x8_t __attribute__((__always_inline__)) lshift_array_vals(const int16x8_t &a)
    {
        return vshlq_n_s16(a, shift_n);
    }
    #endif

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
    #ifdef __SSE3__
    inline __m128i_uint8x16 __attribute__((__always_inline__)) array_pack_16to8(const __m128i_int16x8 &a, const __m128i_int16x8 &b)
    {
        return __m128i_uint8x16{_mm_packus_epi16(a.val, b.val)};
    }
    #endif
    #ifdef __ARM_NEON__
    inline uint8x16_t __attribute__((__always_inline__)) array_pack_16to8(const int16x8_t &a, const int16x8_t &b)
    {
        return vcombine_u8(vqmovun_s16(a), vqmovun_s16(b));
    }
    #endif

template<size_t N>
unpacked16arrays_t<N/2> array_unpack_8to16(const std::array<uint8_t, N> &a)
{
    unpacked16arrays_t<N/2> res;

    for (size_t i=0; i<N/2; i++)
    {
        res.hi[i] = a[i];
    }
    for (size_t i=0; i<N/2; i++)
    {
        res.lo[i] = a[N/2+i];
    }
    return res;
}
    #ifdef __SSE3__
    inline __m128i_pair __attribute__((__always_inline__)) array_unpack_8to16(const __m128i_uint8x16 a)
    {
        __m128i_pair res;
        __m128i zero = _mm_setzero_si128();
        res.hi.val = _mm_unpacklo_epi8(a.val, zero);
        res.lo.val = _mm_unpackhi_epi8(a.val, zero);
        return res;
    }
    #endif
    #ifdef __ARM_NEON__
    inline int16x8_pair_t __attribute__((__always_inline__)) array_unpack_8to16(const uint8x16_t a)
    {
        int16x8_pair_t res;
        res.hi = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
        res.lo = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
        return res;
    }
    #endif

template<class T>
T filled_array8(int16_t val)
{
    T array = {val, val, val, val, val, val, val, val};
    return array;
}

    #ifdef __SSE3__
    template<>
    inline __m128i_int16x8 __attribute__((__always_inline__)) filled_array8<__m128i_int16x8>(int16_t val)
    {
        return __m128i_int16x8{_mm_set1_epi16(val)};
    }
    #endif
    #ifdef __ARM_NEON__
    template<>
    inline int16x8_t __attribute__((__always_inline__)) filled_array8<int16x8_t>(int16_t val)
    {
        return vmovq_n_s16(val);
    }
    #endif

template<class T>
T load_array(const T *ptr)
{
    return *ptr;
}
    #ifdef __SSE3__
    template<>
    inline __m128i_int16x8 __attribute__((__always_inline__)) load_array<__m128i_int16x8>(const __m128i_int16x8 *ptr)
    {
        return __m128i_int16x8{_mm_loadu_si128((const __m128i*)ptr)};
    }
    #endif
    #ifdef __ARM_NEON__
    template<>
    inline int16x8_t __attribute__((__always_inline__)) load_array<int16x8_t>(const int16x8_t *ptr)
    {
        return vld1q_s16((const int16_t*)ptr);
    }
    #endif

template<class T>
T load_aligned_array(const T *ptr)
{
    return *ptr;
}
    #ifdef __SSE3__
    template<>
    inline __m128i_int16x8 __attribute__((__always_inline__)) load_aligned_array<__m128i_int16x8>(const __m128i_int16x8 *ptr)
    {
        return __m128i_int16x8{_mm_load_si128((const __m128i*)ptr)};
    }
    #endif
    #ifdef __ARM_NEON__
    template<>
    inline int16x8_t __attribute__((__always_inline__)) load_aligned_array<int16x8_t>(const int16x8_t *ptr)
    {
        return vld1q_s16((const int16_t*)ptr);
    }
    #endif

template<class T>
void store_array(T *const ptr, const T &val)
{
    *ptr = val;
}
    #ifdef __SSE3__
    template<>
    inline void __attribute__((__always_inline__)) store_array<__m128i_int16x8>(__m128i_int16x8 *const ptr, const __m128i_int16x8 &val)
    {
        _mm_storeu_si128((__m128i*)ptr, val.val);
    }
    template<>
    inline void __attribute__((__always_inline__)) store_array<__m128i_uint8x16>(__m128i_uint8x16 *const ptr, const __m128i_uint8x16 &val)
    {
        _mm_storeu_si128((__m128i*)ptr, val.val);
    }
    #endif
    #ifdef __ARM_NEON__
    template<>
    inline void __attribute__((__always_inline__)) store_array<int16x8_t>(int16x8_t *const ptr, const int16x8_t &val)
    {
        vst1q_s16((int16_t*)ptr, val);
    }
    template<>
    inline void __attribute__((__always_inline__)) store_array<uint8x16_t>(uint8x16_t *const ptr, const uint8x16_t &val)
    {
        vst1q_u8((uint8_t*)ptr, val);
    }
    #endif

template <size_t N>
int32_t sad_array(const std::array<uint8_t, N> &a, const std::array<uint8_t, N> &b)
{
    auto abs_diff = [a, b](unsigned i) -> uint16_t {return a[i] > b[i] ? a[i]-b[i] : b[i]-a[i];};
    int sum = 0;
    for (unsigned i=0; i<N; i++)
    {
        sum += abs_diff(i);
    }
    return sum;
}
    #ifdef __SSE3__
    inline int32_t __attribute__((__always_inline__)) sad_array(const __m128i_uint8x16 &a, const __m128i_uint8x16 &b)
    {
        __m128i sad1 = _mm_sad_epu8 (a.val, b.val);
        return _mm_extract_epi16(sad1,0) + _mm_extract_epi16(sad1,4);
    }
    #endif
    #ifdef __ARM_NEON__
    inline int32_t __attribute__((__always_inline__)) sad_array(const uint8x16_t &a, const uint8x16_t &b)
    {
        const uint8x16_t abs_diff = vabdq_u8 (a,b);
        const uint16x8_t sum1 = vpaddlq_u8(abs_diff);
        const uint32x4_t sum2 = vpaddlq_u16(sum1);
        const uint32x4_t sum3 = vreinterpretq_u32_u64(vpaddlq_u32(sum2));
        return vgetq_lane_u32(sum3,0) + vgetq_lane_u32(sum3,2);
    }
    #endif

template <size_t N>
int32_t sad_array(const std::array<uint8_t, N> &a0, const std::array<uint8_t, N> &a1, const std::array<uint8_t, N> &b0, const std::array<uint8_t, N> &b1)
{
    const auto abs_diff = [](uint8_t a, uint8_t b) -> uint16_t {return a > b ? a - b : b - a;};
    int sum = 0;
    for (unsigned i=0; i<N; i++)
    {
        sum += abs_diff(a0[i], b0[i]);
        sum += abs_diff(a1[i], b1[i]);
    }
    return sum;
}
    #ifdef __SSE3__
    inline int32_t __attribute__((__always_inline__)) sad_array(const __m128i_uint8x16 &a0, const __m128i_uint8x16 &a1, const __m128i_uint8x16 &b0, const __m128i_uint8x16 &b1)
    {
        __m128i sad1 = _mm_sad_epu8 (a0.val, b0.val);
        __m128i sad2 = _mm_sad_epu8 (a1.val, b1.val);
        __m128i sad3 = _mm_add_epi16(sad1,sad2);
        return _mm_extract_epi16(sad3,0) + _mm_extract_epi16(sad3,4);
    }
    #endif
    #ifdef __ARM_NEON__
    inline int32_t __attribute__((__always_inline__)) sad_array(const uint8x16_t &a0, const uint8x16_t &a1, const uint8x16_t &b0, const uint8x16_t &b1)
    {
        const uint8x16_t abs_diff1 = vabdq_u8(a0,b0);
        const uint16x8_t sum1 = vpaddlq_u8(abs_diff1);
        const uint16x8_t sum2 = vabal_u8(sum1, vget_low_u8(a1), vget_low_u8(b1));
        const uint16x8_t sum3 = vabal_u8(sum2, vget_high_u8(a1), vget_high_u8(b1));
        const uint32x4_t sum4 = vpaddlq_u16(sum3);
        const uint32x4_t sum5 = vreinterpretq_u32_u64(vpaddlq_u32(sum4));
        return vgetq_lane_u32(sum5,0) + vgetq_lane_u32(sum5,2);
    }
    #endif

}

#endif // SIMD_HH
