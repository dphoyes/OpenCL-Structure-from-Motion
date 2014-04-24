#ifndef VISO_MONO_CL_H
#define VISO_MONO_CL_H

#include <memory>
#include "viso_mono.h"
#include "opencl_container.hh"

class VisualOdometryMono_CL : public VisualOdometryMono
{
private:
    const std::shared_ptr<OpenCLContainer> cl_container;

    virtual std::vector<int32_t> getInlier (std::vector<Matcher::p_match> &p_matched,Matrix &F);

public:
    VisualOdometryMono_CL (parameters param, std::shared_ptr<OpenCLContainer> cl_container)
        :
            VisualOdometryMono (param)
        ,   cl_container (cl_container)
    {}

};

#endif // VISO_MONO_CL_H

