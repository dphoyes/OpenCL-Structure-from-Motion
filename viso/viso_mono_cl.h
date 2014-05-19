#ifndef VISO_MONO_CL_H
#define VISO_MONO_CL_H

#include <memory>
#include "viso_mono.h"
#include "opencl_container.hh"

class VisualOdometryMono_CL : public VisualOdometryMono
{
private:
    const std::shared_ptr<OpenCLContainer> cl_container;

    cl::Kernel kernel_get_inlier;

    cl::Event copy_inlier_mask_event;

    OpenCLContainer::Buffer buff_inlier_mask;
    OpenCLContainer::Buffer buff_best_inlier_mask;
    OpenCLContainer::Buffer buff_fund_mat;
    OpenCLContainer::Buffer buff_counts;

    size_t work_group_size;
    size_t n_work_groups;
    size_t global_size;

    Matrix                       ransacEstimateF(const std::vector<Matcher::p_match> &p_matched) override;
    uint32_t get_inlier_count(Matrix &F);

public:
    VisualOdometryMono_CL (parameters param, std::shared_ptr<OpenCLContainer> cl_container)
        :   VisualOdometryMono (param)
        ,   cl_container (cl_container)
        ,   kernel_get_inlier (cl_container->getKernel("inlier.cl", "find_inliers"))
    {}

};

#endif // VISO_MONO_CL_H

