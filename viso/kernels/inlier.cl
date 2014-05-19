#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void find_inliers(
        __global const float *match_u1p,      // 0
        __global const float *match_v1p,      // 1
        __global const float *match_u1c,      // 2
        __global const float *match_v1c,      // 3
        __global const double *fund_mat,      // 4
        __global uchar *inlier_mask,          // 5
        double thresh,                        // 6
        uint p_matched_size,                  // 7
        __global ushort *counts               // 8
        )
{
    uint x = get_global_id(0);

    if (x < p_matched_size)
    {
        // extract fundamental matrix
        float f00 = fund_mat[0*3+0]; float f01 = fund_mat[0*3+1]; float f02 = fund_mat[0*3+2];
        float f10 = fund_mat[1*3+0]; float f11 = fund_mat[1*3+1]; float f12 = fund_mat[1*3+2];
        float f20 = fund_mat[2*3+0]; float f21 = fund_mat[2*3+1]; float f22 = fund_mat[2*3+2];

        // extract matches
        float u1 = match_u1p[x];
        float v1 = match_v1p[x];
        float u2 = match_u1c[x];
        float v2 = match_v1c[x];

        // F*x1
        float Fx1u = f00*u1+f01*v1+f02;
        float Fx1v = f10*u1+f11*v1+f12;
        float Fx1w = f20*u1+f21*v1+f22;

        // F'*x2
        float Ftx2u = f00*u2+f10*v2+f20;
        float Ftx2v = f01*u2+f11*v2+f21;

        // x2'*F*x1
        float x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

        // sampson distance
        float d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

        // check threshold
        inlier_mask[x] = fabs(d) < thresh;
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(0) == 0)
    {
        const uint start_i = get_global_id(0);
        const uint end_i = min(convert_uint(get_global_id(0) + get_local_size(0)), p_matched_size);

        uint count = 0;
        for (uint i=start_i; i < end_i; i++)
        {
            count += inlier_mask[i];
        }

        counts[get_group_id(0)] = count;
    }
}
