#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define simd_type double2
#define SIMD_WIDTH (sizeof(simd_type)/sizeof(double))


#ifdef ALTERA_CL
 __attribute__((task))
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void plane_find_best_idx(
        __global const double * restrict d,
        const uint d_len,
        const double threshold,
        const double weight,
        __global uint * restrict return_best_idx
    )
{
    double best_sum = 0;
    uint best_idx = 0;

    for (uint j=0; j<d_len; j++)
    {
        const double d_j = d[j];
        const bool active = d_j > threshold;
        double sum = 0;
        for (uint i=0, ii=0; i<d_len; i+=SIMD_WIDTH, ii++)
        {
            const simd_type dist = d_j - ((global const simd_type*)(d))[ii];
            const simd_type val = exp(-dist*dist*weight);

            double sub_sum = 0;
            #pragma unroll
            for (uint s=0; s<SIMD_WIDTH; s++)
            {
                sub_sum += (i+s < d_len) ? val[s] : 0;
            }

            sum += sub_sum;
        }
        if (!active) sum = 0;

        if (sum>best_sum)
        {
            best_sum = sum;
            best_idx = j;
        }
    }

    *return_best_idx = best_idx;
}
