#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128
#define simd_type double4
#define SIMD_WIDTH (sizeof(simd_type)/sizeof(double))


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_sums(
        __global const double * restrict d,
        const uint d_len,
        const double threshold,
        const double weight,
        __global double * restrict sums
    )
{
    const uint gid0 = get_global_id(0);
    const double d_gid0 = d[gid0];
    const bool active = d_gid0 > threshold;
    double sum = 0;
    for (uint i=0, ii=0; i<d_len; i+=SIMD_WIDTH, ii++)
    {
        const simd_type dist = d_gid0 - ((global const simd_type*)(d))[ii];
        const simd_type val = exp(-dist*dist*weight);

        double sub_sum = val.s0;
        sub_sum += (i+1 < d_len) ? val.s1 : 0;
        sub_sum += (i+2 < d_len) ? val.s2 : 0;
        sub_sum += (i+3 < d_len) ? val.s3 : 0;

        sum += sub_sum;
    }
    sums[gid0] = active ? sum : 0;
}
