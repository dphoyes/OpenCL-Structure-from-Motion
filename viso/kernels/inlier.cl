#define WORK_GROUP_SIZE 128

struct mat_t
{
    float val[3][3];
};

#ifdef ALTERA_CL
__attribute__((task))
#endif
__kernel void find_inliers(
        const uint n_matches,
        const uint iters_per_batch,
        __global const float * restrict match_u1p,
        __global const float * restrict match_v1p,
        __global const float * restrict match_u1c,
        __global const float * restrict match_v1c,
        const float thresh,
        __global const struct mat_t * restrict fund_mats,
        __global uchar * restrict inlier_mask,
        __global ushort * restrict counts,
        __global uchar * restrict best_inliers,
        __global ushort * restrict prev_best_count
    )
{
    for (uint iter=0; iter<iters_per_batch; iter++)
    {
        // extract fundamental matrix
        const struct mat_t f = fund_mats[iter];

        ushort n_inliers = 0;

        for (uint match_id=0; match_id<n_matches; match_id++)
        {
            // extract matches
            float u1 = match_u1p[ match_id ];
            float v1 = match_v1p[ match_id ];
            float u2 = match_u1c[ match_id ];
            float v2 = match_v1c[ match_id ];

            // F*x1
            float Fx1u = f.val[0][0]*u1 + f.val[0][1]*v1 + f.val[0][2];
            float Fx1v = f.val[1][0]*u1 + f.val[1][1]*v1 + f.val[1][2];
            float Fx1w = f.val[2][0]*u1 + f.val[2][1]*v1 + f.val[2][2];

            // F'*x2
            float Ftx2u = f.val[0][0]*u2 + f.val[1][0]*v2 + f.val[2][0];
            float Ftx2v = f.val[0][1]*u2 + f.val[1][1]*v2 + f.val[2][1];

            // x2'*F*x1
            float x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

            // sampson distance
            float d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

            // check threshold
            bool is_inlier = fabs(d) < thresh;
            inlier_mask[iter*n_matches + match_id] = is_inlier;
            if (is_inlier) n_inliers++;
        }

        counts[iter] = n_inliers;
    }

    ushort best_count = *prev_best_count;
    int best_iter = -1;

    for (uint i=0; i<iters_per_batch; i++)
    {
        const ushort iter_count = counts[i];
        if (iter_count > best_count)
        {
            best_iter = i;
            best_count = iter_count;
        }
    }

    if (best_iter != -1)
    {
        *prev_best_count = best_count;
        for (uint i=0; i<n_matches; i++)
        {
            best_inliers[i] = inlier_mask[best_iter*n_matches+i];
        }
    }
}