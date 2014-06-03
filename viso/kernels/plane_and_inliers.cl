#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128


#ifdef ALTERA_CL
__attribute__((num_simd_work_items(2)))
#endif
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
    const uint stride = get_global_size(0);
    const double d_gid0 = d[gid0];
    const bool active = d_gid0 > threshold;
    double sum = 0;
    for (uint i=0; i<d_len; i++)
    {
        const double dist = d_gid0 - d[i];
        const double val = exp(-dist*dist*weight);
        sum += val;
    }
    sums[gid0] = active ? sum : 0;
}
