#ifndef OPENCLCONTAINER_H
#define OPENCLCONTAINER_H

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <fstream>
#include <string>
#include <unordered_map>


class OpenCLContainer
{
public:
    cl::Context context;

    std::vector<cl::Platform> platforms;
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;

    cl::CommandQueue queue;
    std::unordered_map<std::string,cl::Program> programs;

    const cl_device_type DEVICE_TYPE;
    const unsigned N_WORK_GROUPS;
    const unsigned WORK_GROUP_SIZE;
    const unsigned N_WORK_ITEMS;

    class Buffer
    {
    public:
        cl::Buffer buff;
        size_t size;

        Buffer(){}

        Buffer(
            const cl::Context& context,
            cl_mem_flags flags,
            ::size_t size
        ):
            buff(context, flags, size),
            size(size)
        {}
    };

     OpenCLContainer(cl_device_type device_type, unsigned n_work_groups, unsigned work_group_size)
         :   DEVICE_TYPE (device_type)
         ,   N_WORK_GROUPS (n_work_groups)
         ,   WORK_GROUP_SIZE (work_group_size)
         ,   N_WORK_ITEMS (n_work_groups * work_group_size)
     {}

    void init(const std::unordered_map<std::string,std::string> &program_srcs)
    {
        getDevice();

        context = cl::Context(devices);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        for (auto &prog_src: program_srcs)
        {
            makeProgram(prog_src.first, prog_src.second);
        }
    }

    void init(const std::unordered_map<std::string, std::vector<unsigned char> > &program_binaries)
    {
        getDevice();

        context = cl::Context(devices);
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        for (auto &prog_bin: program_binaries)
        {
            makeProgram(prog_bin.first, prog_bin.second);
        }
    }

    void getDevice()
    {
        // Get list of platforms
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");
        std::cerr << "Found " << platforms.size() << " platforms\n";

        for(auto p : platforms)
        {
            std::cerr << "Platform: " << p.getInfo<CL_PLATFORM_VENDOR>() << "\n";

            // Get list of GPU devices
            try
            {
                p.getDevices(DEVICE_TYPE, &devices);
            }
            catch (cl::Error &e) {}

            if (!devices.empty())
            {
                std::cerr << "Found " << devices.size() << " devices\n";

                // Select device
                device = devices.at(0);
                platform = p;
                break;
            }
        }

        try
        {
            std::cerr << "Using: " << platform.getInfo<CL_PLATFORM_VENDOR>() << ", " << device.getInfo<CL_DEVICE_NAME>() << "\n";
        }
        catch (cl::Error &e)
        {
            throw std::runtime_error("No device found.\n");
        }
    }

    void makeProgram(const std::string &program_name, const std::string &program_source)
    {
        const std::string device_id = (DEVICE_TYPE==CL_DEVICE_TYPE_GPU)? "0" : "1";
        const std::string kernel_source = "#define DEVICE_ID " + device_id + "\n" + program_source;

        cl::Program::Sources sources {std::make_pair(kernel_source.c_str(), 0)};

        auto res = programs.emplace(program_name, cl::Program(context, sources));
        auto &prog = res.first->second;

        try
        {
            prog.build(devices);
        }
        catch(...)
        {
            for (auto d : devices)
            {
                std::cerr << "Log for device " << d.getInfo<CL_DEVICE_NAME>() << ":\n\n";
                std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d) << "\n\n";
            }
            throw;
        }
    }

    void makeProgram(const std::string &program_name, const std::vector<unsigned char> &program_bin)
    {
        cl::Program::Binaries binaries {std::make_pair(program_bin.data(), program_bin.size())};

        auto res = programs.emplace(program_name, cl::Program(context, {device}, binaries));
        auto &prog = res.first->second;

        try
        {
            prog.build(devices);
        }
        catch(...)
        {
            for (auto d : devices)
            {
                std::cerr << "Log for device " << d.getInfo<CL_DEVICE_NAME>() << ":\n\n";
                std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d) << "\n\n";
            }
            throw;
        }
    }

    cl::Kernel getKernel(const std::string &file_name, const std::string &kernel_name)
    {
        return cl::Kernel(programs[file_name], kernel_name.c_str());
    }

    cl::Event writeToBuffer(const void *data, Buffer &buff)
    {
        cl::Event ev;
        queue.enqueueWriteBuffer(buff.buff, CL_FALSE, 0, buff.size, data, NULL, &ev);
        return ev;
    }

    long durationOfEvent(const cl::Event &event) const
    {
        cl_ulong start;
        cl_ulong end;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        return end - start;
    }
};

class GPUOpenCLContainer: public OpenCLContainer
{
public:
    GPUOpenCLContainer() : OpenCLContainer {
                                 CL_DEVICE_TYPE_GPU // DEVICE_TYPE
                              ,  2048 // N_WORK_GROUPS
                              ,  256 // WORK_GROUP_SIZE
                          }
    {}
};

class CPUOpenCLContainer: public OpenCLContainer
{
public:
    CPUOpenCLContainer() : OpenCLContainer {
                                 CL_DEVICE_TYPE_CPU // DEVICE_TYPE
                              ,  2048 // N_WORK_GROUPS
                              ,  256 // WORK_GROUP_SIZE
                          }
    {}
};

class FPGAOpenCLContainer: public OpenCLContainer
{
public:
    FPGAOpenCLContainer() : OpenCLContainer {
                                  CL_DEVICE_TYPE_ACCELERATOR // DEVICE_TYPE
                               ,  2048 // N_WORK_GROUPS
                               ,  256 // WORK_GROUP_SIZE
}
    {}
};

#endif // OPENCLCONTAINER_H
