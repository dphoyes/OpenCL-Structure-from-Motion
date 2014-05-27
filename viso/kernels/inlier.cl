#define WORK_GROUP_SIZE 128

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void find_inliers(
        __global const float * restrict match_u1p,
        __global const float * restrict match_v1p,
        __global const float * restrict match_u1c,
        __global const float * restrict match_v1c,
        const uint p_matched_size,
        const uint work_items_per_F,
        const float thresh,
        __global const float * restrict fund_mat,
        __global uchar * restrict inlier_mask
    )
{
    __local float f[9];
    const size_t sub_iter_id = get_global_id(0)%(work_items_per_F);

    if (get_local_id(0) < 9)
    {
        const size_t iter_id = get_global_id(0)/work_items_per_F;
        f[get_local_id(0)] = fund_mat[iter_id*9 + get_local_id(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (sub_iter_id < p_matched_size)
    {
        // extract fundamental matrix
        float f00 = f[0]; float f01 = f[1]; float f02 = f[2];
        float f10 = f[3]; float f11 = f[4]; float f12 = f[5];
        float f20 = f[6]; float f21 = f[7]; float f22 = f[8];

        // extract matches
        float u1 = match_u1p[sub_iter_id];
        float v1 = match_v1p[sub_iter_id];
        float u2 = match_u1c[sub_iter_id];
        float v2 = match_v1c[sub_iter_id];

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

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void sum(
        __global const uchar * restrict in,
        __global ushort * restrict out,
        const uint iter_len,
        const uint batch_width
    )
{
    __local ushort tmp[WORK_GROUP_SIZE];

    const size_t iter_id = get_group_id(0);
    const unsigned base_offset = iter_id*batch_width;

    ushort sum = 0;
    for (unsigned i=get_local_id(0); i<iter_len; i+=get_local_size(0))
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

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void update_best_inliers(
        __global const uchar * restrict inliers,
        __global const ushort * restrict counts,
        const uint iters_per_batch,
        const uint p_matched_size,
        const uint batch_width,
        __global uchar * restrict best_inliers,
        __global ushort * restrict local_best_count
    )
{
    ushort best_count = local_best_count[get_group_id(0)];

    int best_iter = -1;
    for (unsigned i=0; i<iters_per_batch; i++)
    {
        const ushort iter_count = counts[i];
        if (iter_count > best_count)
        {
            best_iter = i;
            best_count = iter_count;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (get_global_id(0) < p_matched_size)
    {
        if (best_iter != -1)
        {
            best_inliers[get_global_id(0)] = inliers[best_iter*batch_width+get_global_id(0)];
            if (get_local_id(0) == 0) local_best_count[get_group_id(0)] = best_count;
        }
    }
}
