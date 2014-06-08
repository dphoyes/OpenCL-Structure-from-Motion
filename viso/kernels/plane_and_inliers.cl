#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128
#define SIMD_WIDTH 4


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
    for (uint i=0; i<d_len; i+=SIMD_WIDTH)
    {
        double sub_sum = 0;
        #pragma unroll
        for (uint s=0; s<SIMD_WIDTH; s++)
        {
            const double dist = d_gid0 - d[i+s];
            const double val = exp(-dist*dist*weight);
            sub_sum += (i+s < d_len) ? val : 0;
        }

        sum += sub_sum;
    }
    sums[gid0] = active ? sum : 0;
}
