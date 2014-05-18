
#include <iostream>
#include <random>
#include "assert.h"
#include "simd.hh"

using namespace simd;

array_8xint16_t to_simd(const std::array<int16_t, 8> &a)
{
    return *(array_8xint16_t*)(&a);
}
array_16xuint8_t to_simd(const std::array<uint8_t, 16> &a)
{
    return *(array_16xuint8_t*)(&a);
}

std::array<int16_t, 8> to_array16(const array_8xint16_t &a)
{
    return *(std::array<int16_t, 8>*)(&a);
}
std::array<uint8_t, 16> to_array8(const array_16xuint8_t &a)
{
    return *(std::array<uint8_t, 16>*)(&a);
}

std::default_random_engine rand_eng;

template <class T>
T rand_array();
template <>
std::array<uint8_t, 16> rand_array<std::array<uint8_t, 16> >()
{
    std::uniform_int_distribution<uint8_t> gen8;
    std::array<uint8_t, 16> arr;
    for (auto &i : arr) i = gen8(rand_eng);
    return arr;
}
template <>
std::array<int16_t, 8> rand_array<std::array<int16_t, 8> >()
{
    std::array<int16_t, 8> arr;
    std::uniform_int_distribution<int16_t> gen16;
    for (auto &i : arr) i = gen16(rand_eng);
    return arr;
}


int main()
{

    for (unsigned i=0; i<10; i++)
    {
        const auto array8_0 = rand_array<std::array<uint8_t, 16> >();
        const auto array8_1 = rand_array<std::array<uint8_t, 16> >();
        const auto array8_2 = rand_array<std::array<uint8_t, 16> >();
        const auto array8_3 = rand_array<std::array<uint8_t, 16> >();
        const auto array16_0 = rand_array<std::array<int16_t, 8> >();
        const auto array16_1 = rand_array<std::array<int16_t, 8> >();
        const auto array16_2 = rand_array<std::array<int16_t, 8> >();

        const array_16xuint8_t array8_0s = to_simd(array8_0);
        const array_16xuint8_t array8_1s = to_simd(array8_1);
        const array_16xuint8_t array8_2s = to_simd(array8_2);
        const array_16xuint8_t array8_3s = to_simd(array8_3);
        const array_8xint16_t array16_0s = to_simd(array16_0);
        const array_8xint16_t array16_1s = to_simd(array16_1);
        const array_8xint16_t array16_2s = to_simd(array16_2);

        assert(
                    add_array_vals<8>(array16_0, array16_1) ==
         to_array16(add_array_vals<8>(array16_0s, array16_1s))
        );
        assert(
                    sub_array_vals<8>(array16_0, array16_1) ==
         to_array16(sub_array_vals<8>(array16_0s, array16_1s))
        );

        assert(
                    rshift_array_vals<8>(array16_0, 2) ==
         to_array16(rshift_array_vals<8>(array16_0s, 2))
        );
        assert(
                    lshift_array_vals<8>(array16_0, 2) ==
         to_array16(lshift_array_vals<8>(array16_0s, 2))
        );

        assert(
                    array_pack_16to8<8>(array16_0, array16_1) ==
         to_array8(array_pack_16to8<8>(array16_0s, array16_1s))
        );

        auto unpacked = array_unpack_8to16<16>(array8_0);
        auto unpackeds = array_unpack_8to16<16>(array8_0s);

        assert(unpacked.hi == to_array16(unpackeds.hi));
        assert(unpacked.lo == to_array16(unpackeds.lo));

        auto mul = mul_array_vals<8>(unpacked, array16_2);
        auto muls = mul_array_vals<8>(unpackeds, array16_2s);

        assert(mul.hi == to_array16(muls.hi));
        assert(mul.lo == to_array16(muls.lo));

        assert(
                   (filled_array8<std::array<int16_t, 8> >(128)) ==
         to_array16(filled_array8<array_8xint16_t>(128))
        );

        assert(
                   load_array(&array16_0) ==
         to_array16(load_array(&array16_0s))
        );
        assert(
                   load_aligned_array(&array16_0) ==
         to_array16(load_aligned_array(&array16_0s))
        );

        std::array<int16_t, 8> store_loc16;
        std::array<uint8_t, 16> store_loc8;
        array_8xint16_t store_loc16s;
        array_16xuint8_t store_loc8s;
        store_array(&store_loc8, array8_0);
        store_array(&store_loc16, array16_0);
        store_array(&store_loc8s, array8_0s);
        store_array(&store_loc16s, array16_0s);
        assert(store_loc16 == to_array16(store_loc16s));
        assert(store_loc8 == to_array8(store_loc8s));

        assert(sad_array<16>(array8_0, array8_1) == sad_array<16>(array8_0s, array8_1s));
        assert(sad_array<16>(array8_0, array8_1, array8_2, array8_3) == sad_array<16>(array8_0s, array8_1s, array8_2s, array8_3s));

    }

    std::cout << "SUCCESS" << std::endl;
}
