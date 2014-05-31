#define WORK_GROUP_SIZE 128
#define MAX_N_MATCHES 5000

struct mat_t
{
    float val[3][3];
};

struct match_t
{
    float u1p;
    float v1p;
    float u1c;
    float v1c;
};

#ifdef ALTERA_CL
__attribute__((task))
#endif
__kernel void find_inliers(
        const uint n_matches,
        const uint iters_per_batch,
        __global const struct match_t * restrict matches,
        __global const struct mat_t * restrict fund_mats,
        const float thresh,
        __global uchar * restrict return_best_inliers,
        __global ushort * restrict prev_best_count
    )
{
    bool inlier_masks[2][MAX_N_MATCHES];
    ushort best_count = *prev_best_count;
    bool found_better_inliers = false;
    bool inlier_scratch = false;

    for (uint iter=0; iter<iters_per_batch; iter++)
    {
        // extract fundamental matrix
        const struct mat_t f = fund_mats[iter];

        ushort n_inliers = 0;

        for (uint match_id=0; match_id<n_matches; match_id++)
        {
            // extract matches
            const struct match_t match = matches[ match_id ];
            const float u1 = match.u1p;
            const float v1 = match.v1p;
            const float u2 = match.u1c;
            const float v2 = match.v1c;

            // F*x1
            const float Fx1u = f.val[0][0]*u1 + f.val[0][1]*v1 + f.val[0][2];
            const float Fx1v = f.val[1][0]*u1 + f.val[1][1]*v1 + f.val[1][2];
            const float Fx1w = f.val[2][0]*u1 + f.val[2][1]*v1 + f.val[2][2];

            // F'*x2
            const float Ftx2u = f.val[0][0]*u2 + f.val[1][0]*v2 + f.val[2][0];
            const float Ftx2v = f.val[0][1]*u2 + f.val[1][1]*v2 + f.val[2][1];

            // x2'*F*x1
            const float x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

            // sampson distance
            const float d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

            // check threshold
            const bool is_inlier = fabs(d) < thresh;

            if (is_inlier) n_inliers++;
            inlier_masks[inlier_scratch][match_id] = is_inlier;
        }

        if (n_inliers > best_count)
        {
            found_better_inliers = true;
            best_count = n_inliers;
            inlier_scratch = !inlier_scratch;
        }
    }

    if (found_better_inliers)
    {
        *prev_best_count = best_count;
        for (uint i=0; i<n_matches; i++)
        {
            return_best_inliers[i] = inlier_masks[!inlier_scratch][i];
        }
    }
}
