#include <viso_mono_cl.h>

using namespace std;

Matrix VisualOdometryMono_CL::ransacEstimateF(const vector<Matcher::p_match> &p_matched)
{
    OpenCLContainer::Buffer buff_p_matched (cl_container->context, CL_MEM_READ_ONLY, p_matched.size()*sizeof(Matcher::p_match));

    kernel_get_inlier.setArg(0, buff_p_matched.buff);
    kernel_get_inlier.setArg(3, param.inlier_threshold);

    cl_container->queue.enqueueWriteBuffer(buff_p_matched.buff, CL_FALSE, 0, buff_p_matched.size, p_matched.data(), NULL, &p_matched_write_event);

    // initial RANSAC estimate of F
    Matrix F;
    inliers.clear();
    for (int32_t k=0;k<param.ransac_iters;k++) {

        // draw random sample set
        vector<int32_t> active = getRandomSample(p_matched.size(),8);

        // estimate fundamental matrix and get inliers
        fundamentalMatrix(p_matched,active,F);
        vector<int32_t> inliers_curr = getInlier(p_matched,F);

        // update model if we are better
        if (inliers_curr.size()>inliers.size())
            inliers = inliers_curr;
    }

    // are there enough inliers?
    if (inliers.size()<10)
    {
        F = Matrix();
    }
    else
    {
        // refine F using all inliers
        fundamentalMatrix(p_matched,inliers,F);
    }

    return F;
}

vector<int32_t> VisualOdometryMono_CL::getInlier (const vector<Matcher::p_match> &p_matched, Matrix &F)
{
    OpenCLContainer::Buffer buff_inlier_mask (cl_container->context, CL_MEM_WRITE_ONLY, p_matched.size()*sizeof(int));
    OpenCLContainer::Buffer buff_fund_mat (cl_container->context, CL_MEM_READ_ONLY, 9*sizeof(double));

    kernel_get_inlier.setArg(1, buff_fund_mat.buff);
    kernel_get_inlier.setArg(2, buff_inlier_mask.buff);

    cl::Event fund_mat_write_event; cl_container->queue.enqueueWriteBuffer(buff_fund_mat.buff, CL_FALSE, 0, buff_fund_mat.size, &F.val[0][0], NULL, &fund_mat_write_event);

    cl::NDRange offset;
    cl::NDRange globalSize (p_matched.size());
    cl::NDRange localSize (cl::NullRange);

    std::vector<cl::Event> kernel_deps {p_matched_write_event, fund_mat_write_event};
    cl::Event get_inlier_complete_event; cl_container->queue.enqueueNDRangeKernel(kernel_get_inlier, offset, globalSize, localSize, &kernel_deps, &get_inlier_complete_event);

    std::vector<int> inlier_mask (p_matched.size());
    std::vector<cl::Event> inlier_mask_read_deps {get_inlier_complete_event};
    cl::Event inlier_mask_read_event; cl_container->queue.enqueueReadBuffer(buff_inlier_mask.buff, CL_FALSE, 0, buff_inlier_mask.size, inlier_mask.data(), &inlier_mask_read_deps, &inlier_mask_read_event);

    std::vector<cl::Event> wait_events {inlier_mask_read_event};
    cl::WaitForEvents(wait_events);

    // vector with inliers
    vector<int32_t> inliers;

    // for all matches do
    for (int32_t i=0; i<(int32_t)inlier_mask.size(); i++)
    {
        if (inlier_mask[i] > 0)
        {
            inliers.push_back(i);
        }
    }

    // return set of all inliers
    return inliers;
}
