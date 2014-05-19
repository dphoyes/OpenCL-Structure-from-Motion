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
    cl::Kernel kernel_sum;
    cl::Kernel kernel_update_inliers;

    OpenCLContainer::Buffer buff_inlier_mask;
    OpenCLContainer::Buffer buff_best_inlier_mask;
    OpenCLContainer::Buffer buff_fund_mat;
    OpenCLContainer::Buffer buff_counts;
    OpenCLContainer::Buffer buff_best_count;

    size_t work_group_size;
    size_t n_work_groups;
    size_t global_size;

    Matrix ransacEstimateF(const std::vector<Matcher::p_match> &p_matched) override;
    void update_best_inliers(Matrix &F);

public:
    VisualOdometryMono_CL (parameters param, std::shared_ptr<OpenCLContainer> cl_container)
        :   VisualOdometryMono (param)
        ,   cl_container (cl_container)
        ,   kernel_get_inlier (cl_container->getKernel("inlier.cl", "find_inliers"))
        ,   kernel_sum (cl_container->getKernel("inlier.cl", "sum"))
        ,   kernel_update_inliers (cl_container->getKernel("inlier.cl", "update_best_inliers"))
    {}

};

#endif // VISO_MONO_CL_H

