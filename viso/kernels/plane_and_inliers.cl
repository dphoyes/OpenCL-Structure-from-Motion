#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define SIMD_WIDTH 2


#ifdef ALTERA_CL
 __attribute__((task))
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void plane_calc_sums(
        __global const double * restrict d,
        const uint d_len,
        const double threshold,
        const double weight,
        __global double * restrict sums
    )
{
    for (uint j=0; j<d_len; j++)
    {
        const double d_j = d[j];
        const bool active = d_j > threshold;
        double sum = 0;
        for (uint i=0; i<d_len; i+=SIMD_WIDTH)
        {
            double sub_sum = 0;
            #pragma unroll
            for (uint s=0; s<SIMD_WIDTH; s++)
            {
                const double dist = d_j - d[i+s];
                const double val = exp(-dist*dist*weight);
                sub_sum += (i+s < d_len) ? val : 0;
            }
            sum += sub_sum;
        }
        sums[j] = active ? sum : 0;
    }
}
