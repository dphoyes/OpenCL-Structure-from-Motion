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

__kernel void kernel_xy(
        __global const struct p_match_t *p_matched, // 0
        __global const double *fund_mat, // 1
        __global int *inlier_mask, // 2
        double thresh // 3
        )
{
    uint x = get_global_id(0);

    // extract fundamental matrix
    double f00 = fund_mat[0*3+0]; double f01 = fund_mat[0*3+1]; double f02 = fund_mat[0*3+2];
    double f10 = fund_mat[1*3+0]; double f11 = fund_mat[1*3+1]; double f12 = fund_mat[1*3+2];
    double f20 = fund_mat[2*3+0]; double f21 = fund_mat[2*3+1]; double f22 = fund_mat[2*3+2];

    // extract matches
    double u1 = p_matched[x].u1p;
    double v1 = p_matched[x].v1p;
    double u2 = p_matched[x].u1c;
    double v2 = p_matched[x].v1c;

    // F*x1
    double Fx1u = f00*u1+f01*v1+f02;
    double Fx1v = f10*u1+f11*v1+f12;
    double Fx1w = f20*u1+f21*v1+f22;

    // F'*x2
    double Ftx2u = f00*u2+f10*v2+f20;
    double Ftx2v = f01*u2+f11*v2+f21;

    // x2'*F*x1
    double x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

    // sampson distance
    double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

    // check threshold
    inlier_mask[x] = fabs(d)<thresh ? 1 : -1;
}
