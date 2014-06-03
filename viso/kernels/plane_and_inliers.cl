#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128


#ifdef ALTERA_CL
__attribute__((num_simd_work_items(2)))
#endif
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_dists(
        __global const double * restrict d,
        const double weight,
        const uint d_len,
        __global double * restrict out
    )
{
    const uint gid0 = get_global_id(0);
    const uint stride = get_global_size(0);
    for (uint i=0; i<d_len; i++)
    {
        const double dist = d[gid0] - d[i];
        const double val = exp(-dist*dist*weight);
        out[i*stride + gid0] = val;
    }
}


#ifdef ALTERA_CL
__attribute__((num_simd_work_items(2)))
#endif
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_sum(
        __global const double * restrict in,
        const uint n_to_sum,
        __global double * restrict sums
    )
{
    const uint gid0 = get_global_id(0);
    const uint stride = get_global_size(0);
    double sum = 0;
    for (uint i=0; i<n_to_sum; i++)
    {
        sum += in[i*stride + gid0];
    }
    sums[gid0] = sum;
}
