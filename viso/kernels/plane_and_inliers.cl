#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_dists(
        __global const double * restrict d,
        const uint stride,
        const double weight,
        __global double * restrict out
    )
{
    uint gid0 = get_global_id(0);
    uint gid1 = get_global_id(1);
    if (gid0 < stride && gid1 < stride)
    {
        double dist = d[gid0] - d[gid1];
        double val = exp(-dist*dist*weight);
        out[gid1*stride + gid0] = val;
    }
}
