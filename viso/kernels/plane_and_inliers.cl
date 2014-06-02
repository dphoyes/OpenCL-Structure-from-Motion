#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 128


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_calc_dists(
        __global const double * restrict d,
        const uint d_len,
        const double weight,
        __global double * restrict out
    )
{
    if (get_global_id(0) < d_len && get_global_id(1) < d_len)
    {
        double dist = d[get_global_id(0)] - d[get_global_id(1)];
        double val = exp(-dist*dist*weight);
        out[get_global_id(1)*d_len+get_global_id(0)] = val;
    }
}
