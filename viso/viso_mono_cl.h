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

    cl::Event p_matched_write_event;


    Matrix                       ransacEstimateF(const std::vector<Matcher::p_match> &p_matched) override;
    virtual std::vector<int32_t> getInlier (const std::vector<Matcher::p_match> &p_matched,Matrix &F) override;

public:
    VisualOdometryMono_CL (parameters param, std::shared_ptr<OpenCLContainer> cl_container)
        :   VisualOdometryMono (param)
        ,   cl_container (cl_container)
        ,   kernel_get_inlier (cl_container->getKernel("inlier.cl", "kernel_xy"))
    {}

};

#endif // VISO_MONO_CL_H

