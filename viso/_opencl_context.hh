#ifndef OPENCL_CONTEXT_HH
#define OPENCL_CONTEXT_HH

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

class OpenClContext
{
public:
    cl::Context context;
    std::vector<cl::Platform> platforms;
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;
    cl::Program program;

    cl::Buffer buffPMatch;
    cl::Buffer buffFundMat;
    cl::Buffer buffInlierMask;

    cl::Kernel kernel;

    cl::CommandQueue queue;

    OpenClContext()
    {
        // Get list of platforms

        cl::Platform::get(&platforms);
        if(platforms.size()==0)
            throw std::runtime_error("No OpenCL platforms found.");

        std::cerr<<"Found "<<platforms.size()<<" platforms\n";
        for(unsigned i=0;i<platforms.size();i++){
            std::string vendor=platforms[i].getInfo<CL_PLATFORM_VENDOR>();
            std::cerr<<"  Platform "<<i<<" : "<<vendor<<"\n";
        }

        // Select platform by env variable
        int selectedPlatform=0;
        if(getenv("HPCE_SELECT_PLATFORM")){
            selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
        }
        std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
        platform=platforms.at(selectedPlatform);


        // Get list of devices
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(devices.size()==0){
            throw std::runtime_error("No opencl devices found.\n");
        }

        std::cerr<<"Found "<<devices.size()<<" devices\n";
        for(unsigned i=0;i<devices.size();i++){
            std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
            std::cerr<<"  Device "<<i<<" : "<<name<<"\n";
        }

        // Select device by env variable
        int selectedDevice=0;
        if(getenv("HPCE_SELECT_DEVICE")){
            selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
        }
        std::cerr<<"Choosing device "<<selectedDevice<<"\n";
        device=devices.at(selectedDevice);

        // Create the context
        context = cl::Context(devices);

        buffPMatch = cl::Buffer(context, CL_MEM_READ_ONLY, 12*16*3000);
        buffFundMat = cl::Buffer(context, CL_MEM_READ_ONLY, 9*8);
        buffInlierMask = cl::Buffer(context, CL_MEM_WRITE_ONLY, 12*4*3000);

        queue = cl::CommandQueue(context, device);

        exit(1);

    }

    void makeKernel(std::string &kernelSource)
    {
        cl::Program::Sources sources;   // A vector of (data,length) pairs
        sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1)); // push on our single string

        program = cl::Program(context, sources);
        try{
            program.build(devices);
        }catch(...){
            for(unsigned i=0;i<devices.size();i++){
                std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
                std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
            }
            throw;
        }

        kernel = cl::Kernel(program, "kernel_xy");
    }
};

#endif // OPENCL_CONTEXT_HH
