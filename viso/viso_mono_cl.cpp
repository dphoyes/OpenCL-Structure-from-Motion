#include <viso_mono_cl.h>

using namespace std;

Matrix VisualOdometryMono_CL::ransacEstimateF(const vector<Matcher::p_match> &p_matched)
{
    work_group_size = 128;
    n_work_groups = (p_matched.size() + work_group_size - 1)/work_group_size;
    global_size = n_work_groups * work_group_size;

    std::vector<float> match_u1p(p_matched.size());
    std::vector<float> match_v1p(p_matched.size());
    std::vector<float> match_u1c(p_matched.size());
    std::vector<float> match_v1c(p_matched.size());

    for (unsigned i=0; i<p_matched.size(); i++)
    {
        match_u1p[i] = p_matched[i].u1p;
        match_v1p[i] = p_matched[i].v1p;
        match_u1c[i] = p_matched[i].u1c;
        match_v1c[i] = p_matched[i].v1c;
    }

    OpenCLContainer::Buffer buff_match_u1p (cl_container->context, CL_MEM_READ_ONLY, p_matched.size()*sizeof(float));
    OpenCLContainer::Buffer buff_match_v1p (cl_container->context, CL_MEM_READ_ONLY, p_matched.size()*sizeof(float));
    OpenCLContainer::Buffer buff_match_u1c (cl_container->context, CL_MEM_READ_ONLY, p_matched.size()*sizeof(float));
    OpenCLContainer::Buffer buff_match_v1c (cl_container->context, CL_MEM_READ_ONLY, p_matched.size()*sizeof(float));

    buff_fund_mat = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_ONLY, 9*sizeof(double));
    buff_inlier_mask = OpenCLContainer::Buffer(cl_container->context, CL_MEM_WRITE_ONLY, p_matched.size()*sizeof(char));
    buff_counts = OpenCLContainer::Buffer(cl_container->context, CL_MEM_WRITE_ONLY, n_work_groups*sizeof(uint16_t));
    buff_best_inlier_mask = OpenCLContainer::Buffer(cl_container->context, CL_MEM_WRITE_ONLY, buff_inlier_mask.size);

    kernel_get_inlier.setArg(0, buff_match_u1p.buff);
    kernel_get_inlier.setArg(1, buff_match_v1p.buff);
    kernel_get_inlier.setArg(2, buff_match_u1c.buff);
    kernel_get_inlier.setArg(3, buff_match_v1c.buff);
    kernel_get_inlier.setArg(4, buff_fund_mat.buff);
    kernel_get_inlier.setArg(5, buff_inlier_mask.buff);
    kernel_get_inlier.setArg(6, float(param.inlier_threshold));
    kernel_get_inlier.setArg(7, uint32_t(p_matched.size()));
    kernel_get_inlier.setArg(8, buff_counts.buff);
    kernel_get_inlier.setArg(9, 9*sizeof(cl_float), nullptr);

    cl::Event match_u1p_write_event = cl_container->writeToBuffer(match_u1p.data(), buff_match_u1p);
    cl::Event match_v1p_write_event = cl_container->writeToBuffer(match_v1p.data(), buff_match_v1p);
    cl::Event match_u1c_write_event = cl_container->writeToBuffer(match_u1c.data(), buff_match_u1c);
    cl::Event match_v1c_write_event = cl_container->writeToBuffer(match_v1c.data(), buff_match_v1c);

    {
        std::vector<cl::Event> wait_events {match_u1p_write_event, match_v1p_write_event, match_u1c_write_event, match_v1c_write_event};
        cl::WaitForEvents(wait_events);
    }
    copy_inlier_mask_event = match_u1p_write_event;

    // initial RANSAC estimate of F
    Matrix F;
    uint32_t best_n_inliers = 0;
    for (int32_t k=0;k<param.ransac_iters;k++) {

        // draw random sample set
        vector<int32_t> active = getRandomSample(p_matched.size(),8);

        // estimate fundamental matrix and get inliers
        fundamentalMatrix(p_matched,active,F);
        uint32_t n_inliers = get_inlier_count(F);

        // update model if we are better
        if (n_inliers > best_n_inliers)
        {
            best_n_inliers = n_inliers;
            cl_container->queue.enqueueCopyBuffer(buff_inlier_mask.buff, buff_best_inlier_mask.buff, 0, 0, buff_inlier_mask.size, NULL, &copy_inlier_mask_event);
        }
    }

    std::vector<uint8_t> inlier_mask (p_matched.size());
    std::vector<cl::Event> inlier_mask_read_deps {copy_inlier_mask_event};
    cl::Event inlier_mask_read_event; cl_container->queue.enqueueReadBuffer(buff_best_inlier_mask.buff, CL_FALSE, 0, buff_best_inlier_mask.size, inlier_mask.data(), &inlier_mask_read_deps, &inlier_mask_read_event);

    {
        std::vector<cl::Event> wait_events {inlier_mask_read_event};
        cl::WaitForEvents(wait_events);
    }

    inliers.clear();

    // for all matches do
    for (int32_t i=0; i<(int32_t)inlier_mask.size(); i++)
    {
        if (inlier_mask[i])
        {
            inliers.push_back(i);
        }
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

uint32_t VisualOdometryMono_CL::get_inlier_count(Matrix &F)
{
    cl::Event fund_mat_write_event = cl_container->writeToBuffer(&F.val[0][0], buff_fund_mat);

    cl::NDRange offset;
    cl::NDRange globalSize (global_size);
    cl::NDRange localSize (work_group_size);

    std::vector<cl::Event> kernel_deps {fund_mat_write_event, copy_inlier_mask_event};
    cl::Event get_inlier_complete_event; cl_container->queue.enqueueNDRangeKernel(kernel_get_inlier, offset, globalSize, localSize, &kernel_deps, &get_inlier_complete_event);

    std::vector<uint16_t> inlier_counts(n_work_groups);
    std::vector<cl::Event> inlier_counts_read_deps {get_inlier_complete_event};
    cl::Event inlier_counts_read_event; cl_container->queue.enqueueReadBuffer(buff_counts.buff, CL_FALSE, 0, buff_counts.size, inlier_counts.data(), &inlier_counts_read_deps, &inlier_counts_read_event);
    {
        std::vector<cl::Event> wait_events {inlier_counts_read_event};
        cl::WaitForEvents(wait_events);
    }

//    std::cerr << "get_inlier_complete_event: " << cl_container->durationOfEvent(get_inlier_complete_event) << "  ";
//    std::cerr << "inlier_counts_read_event: " << cl_container->durationOfEvent(inlier_counts_read_event) << "  ";
//    std::cerr << std::endl;


    uint32_t n_inliers = 0;
    for (auto c : inlier_counts)
    {
        n_inliers += c;
    }

    return n_inliers;
}
