#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128
#define simd_type float4
#define SIMD_WIDTH (sizeof(simd_type)/sizeof(float))


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_sums(
        __global const float * restrict d,
        const uint d_len,
        const float threshold,
        const float weight,
        __global float * restrict sums
    )
{
    const uint gid0 = get_global_id(0);
    const float d_gid0 = d[gid0];
    const bool active = d_gid0 > threshold;
    float sum = 0;
    for (uint i=0, ii=0; i<d_len; i+=SIMD_WIDTH, ii++)
    {
        const simd_type dist = d_gid0 - ((global const simd_type*)(d))[ii];
        const simd_type val = exp(-dist*dist*weight);

        float sub_sum = 0;
        #pragma unroll
        for (uint s=0; s<SIMD_WIDTH; s++)
        {
            sub_sum += (i+s < d_len) ? val[s] : 0;
        }

        sum += sub_sum;
    }
    sums[gid0] = active ? sum : 0;
}
