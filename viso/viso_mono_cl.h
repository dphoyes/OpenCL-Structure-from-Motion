#ifndef VISO_MONO_CL_H
#define VISO_MONO_CL_H

#include <memory>
#include "viso_mono.h"
#include "opencl_wrapper.hh"

class VisualOdometryMono_CL : public VisualOdometryMono
{
private:
    OpenCL::Container &cl_container;

    Matrix ransacEstimateF(const std::vector<Matcher::p_match> &p_matched) override;

public:
    VisualOdometryMono_CL (parameters param, OpenCL::Container &cl_container)
        :   VisualOdometryMono (param)
        ,   cl_container (cl_container)
    {}

};

#endif // VISO_MONO_CL_H

