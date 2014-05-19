#pragma OPENCL EXTENSION cl_khr_fp64 : enable

        struct p_match_t {
          float   u1p,v1p; // u,v-coordinates in previous left  image
          int i1p;     // feature index (for tracking)
          float   u2p,v2p; // u,v-coordinates in previous right image
          int i2p;     // feature index (for tracking)
          float   u1c,v1c; // u,v-coordinates in current  left  image
          int i1c;     // feature index (for tracking)
          float   u2c,v2c; // u,v-coordinates in current  right image
          int i2c;     // feature index (for tracking)
        };

__kernel void find_inliers(
        __global const struct p_match_t *p_matched, // 0
        __global const double *fund_mat, // 1
        __global uchar *inlier_mask, // 2
        double thresh, // 3
        uint p_matched_size, // 4
        __global ushort *counts // 5
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
        float u1 = p_matched[x].u1p;
        float v1 = p_matched[x].v1p;
        float u2 = p_matched[x].u1c;
        float v2 = p_matched[x].v1c;

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
