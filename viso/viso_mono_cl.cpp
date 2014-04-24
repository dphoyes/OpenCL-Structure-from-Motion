#include <viso_mono_cl.h>

using namespace std;

vector<int32_t> VisualOdometryMono_CL::getInlier (vector<Matcher::p_match> &p_matched, Matrix &F)
{
    // extract fundamental matrix
    double f00 = F.val[0][0]; double f01 = F.val[0][1]; double f02 = F.val[0][2];
    double f10 = F.val[1][0]; double f11 = F.val[1][1]; double f12 = F.val[1][2];
    double f20 = F.val[2][0]; double f21 = F.val[2][1]; double f22 = F.val[2][2];

    // loop variables
    double u1,v1,u2,v2;
    double x2tFx1;
    double Fx1u,Fx1v,Fx1w;
    double Ftx2u,Ftx2v;

    // vector with inliers
    vector<int32_t> inliers;

    // for all matches do
    for (int32_t i=0; i<(int32_t)p_matched.size(); i++)
    {

        // extract matches
        u1 = p_matched[i].u1p;
        v1 = p_matched[i].v1p;
        u2 = p_matched[i].u1c;
        v2 = p_matched[i].v1c;

        // F*x1
        Fx1u = f00*u1+f01*v1+f02;
        Fx1v = f10*u1+f11*v1+f12;
        Fx1w = f20*u1+f21*v1+f22;

        // F'*x2
        Ftx2u = f00*u2+f10*v2+f20;
        Ftx2v = f01*u2+f11*v2+f21;

        // x2'*F*x1
        x2tFx1 = u2*Fx1u+v2*Fx1v+Fx1w;

        // sampson distance
        double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);

        // check threshold
        if (fabs(d)<param.inlier_threshold)
        {
            inliers.push_back(i);
        }
    }

    // return set of all inliers
    return inliers;
}
