#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128


__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void plane_calc_dists(
        __global const double * restrict d,
        const uint stride,
        const double weight,
        __global double * restrict out
    )
{
    uint gid0 = get_global_id(0)%stride;
    uint gid1 = get_global_id(0)/stride;
    double dist = d[gid0] - d[gid1];
    double val = exp(-dist*dist*weight);
    out[gid1*stride + gid0] = val;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void plane_sum(
        __global const double * restrict in,
        const uint stride,
        const uint n_to_sum,
        __global double * restrict sums
    )
{
    uint gid0 = get_global_id(0);
    double sum = 0;
    for (uint i=0; i<n_to_sum; i++)
    {
        sum += in[i*stride + gid0];
    }
    sums[gid0] = sum;
}
