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

    buff_fund_mat = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_ONLY, 9*sizeof(cl_double));
    buff_inlier_mask = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_WRITE, p_matched.size()*sizeof(cl_uchar));
    buff_counts = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_WRITE, n_work_groups*sizeof(cl_ushort));
    buff_best_inlier_mask = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_WRITE, p_matched.size()*sizeof(cl_uchar));
    buff_best_count = OpenCLContainer::Buffer(cl_container->context, CL_MEM_READ_WRITE, n_work_groups*sizeof(cl_ushort));

    kernel_get_inlier.setArg(0, buff_match_u1p.buff);
    kernel_get_inlier.setArg(1, buff_match_v1p.buff);
    kernel_get_inlier.setArg(2, buff_match_u1c.buff);
    kernel_get_inlier.setArg(3, buff_match_v1c.buff);
    kernel_get_inlier.setArg(4, cl_uint(p_matched.size()));
    kernel_get_inlier.setArg(5, cl_float(param.inlier_threshold));
    kernel_get_inlier.setArg(6, buff_fund_mat.buff);
    kernel_get_inlier.setArg(7, buff_inlier_mask.buff);
    kernel_get_inlier.setArg(8, 9*sizeof(cl_float), nullptr);

    kernel_sum.setArg(0, buff_inlier_mask.buff);
    kernel_sum.setArg(1, buff_counts.buff);
    kernel_sum.setArg(2, uint32_t(p_matched.size()));
    kernel_sum.setArg(3, work_group_size*sizeof(cl_ushort), nullptr);

    kernel_update_inliers.setArg(0, buff_inlier_mask.buff);
    kernel_update_inliers.setArg(1, buff_counts.buff);
    kernel_update_inliers.setArg(2, cl_uint(n_work_groups));
    kernel_update_inliers.setArg(3, cl_uint(p_matched.size()));
    kernel_update_inliers.setArg(4, buff_best_inlier_mask.buff);
    kernel_update_inliers.setArg(5, buff_best_count.buff);
    kernel_update_inliers.setArg(6, work_group_size*sizeof(cl_ushort), nullptr);

    cl::Event match_u1p_write_event = cl_container->writeToBuffer(match_u1p.data(), buff_match_u1p);
    cl::Event match_v1p_write_event = cl_container->writeToBuffer(match_v1p.data(), buff_match_v1p);
    cl::Event match_u1c_write_event = cl_container->writeToBuffer(match_u1c.data(), buff_match_u1c);
    cl::Event match_v1c_write_event = cl_container->writeToBuffer(match_v1c.data(), buff_match_v1c);

    const std::vector<cl_ushort> zeros(n_work_groups, 0);
    cl::Event init_best_count_event = cl_container->writeToBuffer(zeros.data(), buff_best_count);

    {
        std::vector<cl::Event> wait_events {match_u1p_write_event, match_v1p_write_event, match_u1c_write_event, match_v1c_write_event, init_best_count_event};
        cl::WaitForEvents(wait_events);
    }

    // initial RANSAC estimate of F
    Matrix F;
    for (int32_t k=0; k<param.ransac_iters; k++)
    {
        // draw random sample set
        vector<int32_t> active = getRandomSample(p_matched.size(),8);

        // estimate fundamental matrix and get inliers
        fundamentalMatrix(p_matched,active,F);
        update_best_inliers(F);
    }

    std::vector<uint8_t> inlier_mask (p_matched.size());
    cl::Event inlier_mask_read_event; cl_container->queue.enqueueReadBuffer(buff_best_inlier_mask.buff, CL_FALSE, 0, buff_best_inlier_mask.size, inlier_mask.data(), nullptr, &inlier_mask_read_event);
    inlier_mask_read_event.wait();

    inliers.clear();
    // for all matches do
    for (unsigned i=0; i<inlier_mask.size(); i++)
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

void VisualOdometryMono_CL::update_best_inliers(Matrix &F)
{
    cl::Event fund_mat_write_event = cl_container->writeToBuffer(&F.val[0][0], buff_fund_mat);

    cl::NDRange offset;
    cl::NDRange globalSize (global_size);
    cl::NDRange localSize (work_group_size);

    std::vector<cl::Event> inlier_kernel_deps {fund_mat_write_event};
    cl::Event get_inlier_complete_event; cl_container->queue.enqueueNDRangeKernel(kernel_get_inlier, offset, globalSize, localSize, &inlier_kernel_deps, &get_inlier_complete_event);

    std::vector<cl::Event> sum_kernel_deps {get_inlier_complete_event};
    cl::Event sum_complete_event; cl_container->queue.enqueueNDRangeKernel(kernel_sum, offset, globalSize, localSize, &sum_kernel_deps, &sum_complete_event);

    std::vector<cl::Event> update_kernel_deps {sum_complete_event};
    cl::Event update_complete_event; cl_container->queue.enqueueNDRangeKernel(kernel_update_inliers, offset, globalSize, localSize, &update_kernel_deps, &update_complete_event);

    update_complete_event.wait();

//    std::cerr << "get_inlier_complete_event: " << cl_container->durationOfEvent(get_inlier_complete_event) << "  ";
//    std::cerr << "sum_complete_event: " << cl_container->durationOfEvent(sum_complete_event) << "  ";
//    std::cerr << "update_complete_event: " << cl_container->durationOfEvent(update_complete_event) << "  ";
//    std::cerr << std::endl;
}
