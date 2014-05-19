#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void find_inliers(
        __global const float *match_u1p,   // 0
        __global const float *match_v1p,   // 1
        __global const float *match_u1c,   // 2
        __global const float *match_v1c,   // 3
        const uint p_matched_size,         // 4
        const float thresh,                // 5
        __global const double *fund_mat,   // 6
        __global uchar *inlier_mask,       // 7
        __local float *f                   // 8
    )
{
    if (get_local_id(0) < 9)
    {
        f[get_local_id(0)] = fund_mat[get_local_id(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) < p_matched_size)
    {
        // extract fundamental matrix
        float f00 = f[0]; float f01 = f[1]; float f02 = f[2];
        float f10 = f[3]; float f11 = f[4]; float f12 = f[5];
        float f20 = f[6]; float f21 = f[7]; float f22 = f[8];

        // extract matches
        float u1 = match_u1p[get_global_id(0)];
        float v1 = match_v1p[get_global_id(0)];
        float u2 = match_u1c[get_global_id(0)];
        float v2 = match_v1c[get_global_id(0)];

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
        inlier_mask[get_global_id(0)] = fabs(d) < thresh;
    }
}

__kernel void sum(
        __global const uchar *in,           // 0
        __global ushort *out,               // 1
        const uint len,                    // 2
        __local ushort *tmp               // 3
    )
{
    tmp[get_local_id(0)] = (get_global_id(0) < len) ? in[get_global_id(0)] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0)/2; stride > 0; stride >>= 1)
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

__kernel void update_best_inliers(
        __global const uchar *inliers,  // 0
        __global const ushort *counts,  // 1
        const uint n_counts,            // 2
        const uint p_matched_size,      // 3
        __global uchar *best_inliers,   // 4
        __global ushort *best_count,    // 5
        __local ushort *tmp             // 6
    )
{
    ushort count = 0;
    for (unsigned i=0; i<n_counts; i++)
    {
        count += counts[i];
    }

    if (count > best_count[get_group_id(0)] && get_global_id(0) < p_matched_size)
    {
        best_inliers[get_global_id(0)] = inliers[get_global_id(0)];
    }

    barrier(0);

    if (count > best_count[get_group_id(0)] && get_local_id(0) == 0)
    {
        best_count[get_group_id(0)] = count;
    }
}
