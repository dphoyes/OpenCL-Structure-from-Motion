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
    uint gid0 = get_global_id(0)%stride;
    uint gid1 = get_global_id(0)/stride;
    if (gid0 < stride && gid1 < stride)
    {
        double dist = d[gid0] - d[gid1];
        double val = exp(-dist*dist*weight);
        out[gid1*stride + gid0] = val;
    }
}


__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
__kernel void plane_sum(
        __global const double * restrict in,
        const uint stride,
        const uint n_to_sum,
        __global double * restrict out
    )
{
    __local double tmp[WORK_GROUP_SIZE];

    const uint base_offset = get_group_id(0)*stride;

    double sum = 0;
    for (uint i=get_local_id(0); i<n_to_sum; i+=get_local_size(0))
    {
        sum += in[base_offset+i];
    }

    tmp[get_local_id(0)] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2)
    {
        if (get_local_id(0) < stride)
        {
            tmp[get_local_id(0)] += tmp[get_local_id(0) + stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0)
    {
        out[get_group_id(0)] = tmp[0];
    }
}