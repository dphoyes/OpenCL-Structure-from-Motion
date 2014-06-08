#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128
#define simd_type double2
#define SIMD_WIDTH (sizeof(simd_type)/sizeof(double))

channel double SUM_CHANNEL __attribute__((depth(8)));


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_sums(
        __global const double * restrict d,
        const uint d_len,
        const double threshold,
        const double weight
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

        double sub_sum = 0;
        #pragma unroll
        for (uint s=0; s<SIMD_WIDTH; s++)
        {
            sub_sum += (i+s < d_len) ? val[s] : 0;
        }

        sum += sub_sum;
    }
    write_channel_altera(SUM_CHANNEL, active ? sum : 0);
}


#ifdef ALTERA_CL
 __attribute__((task))
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void find_best_idx(
        const uint d_len,
        const uint cl_d_len,
        __global uint * restrict return_best_idx
    )
{
    double best_sum = 0;
    uint best_idx = 0;

    for (uint i=0; i<cl_d_len; i++)
    {
        const double sum = read_channel_altera(SUM_CHANNEL);
        if (i < d_len && sum > best_sum)
        {
            best_sum = sum;
            best_idx = i;
        }
    }

    *return_best_idx = best_idx;
}
