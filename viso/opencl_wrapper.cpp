#include <iostream>
#include "opencl_wrapper.hh"

namespace OpenCL
{

/*
 * Kernel
 */

Kernel::Kernel(Container &cl_container, const cl::Program& program, const char* name)
    :   cl_container (cl_container)
    ,   kernel (program, name)
    ,   reqd_local_size (kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(cl_container.device))
    ,   local_size (reqd_local_size[0])
{}

cl::Event Kernel::start(const std::vector<cl::Event> &deps)
{
    cl::Event ev;
    cl_container.queue.enqueueNDRangeKernel(kernel, offset, global_size, local_size, &deps, &ev);
    return ev;
}

Kernel& Kernel::setRange(const cl::NDRange &global)
{
    this->global_size = global;
    return *this;
}


/*
 * Container
 */

Container::Container(cl_device_type device_type, unsigned n_work_groups, unsigned work_group_size)
    :   DEVICE_TYPE (device_type)
    ,   N_WORK_GROUPS (n_work_groups)
    ,   WORK_GROUP_SIZE (work_group_size)
    ,   N_WORK_ITEMS (n_work_groups * work_group_size)
{}

GPUContainer::GPUContainer() : Container (CL_DEVICE_TYPE_GPU,  2048,  256) {}
CPUContainer::CPUContainer() : Container (CL_DEVICE_TYPE_CPU,  2048,  256) {}
FPGAContainer::FPGAContainer() : Container (CL_DEVICE_TYPE_ACCELERATOR,  2048,  256) {}

Container::Container(const Container &c) { throw std::runtime_error("OpenCL::Container copy not defined\n"); }

void Container::init(const std::unordered_map<std::string,std::string> &program_srcs)
{
    getDevice();

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    for (auto &prog_src: program_srcs)
    {
        makeProgram(prog_src.first, prog_src.second);
    }
}

void Container::init(const std::unordered_map<std::string, std::vector<unsigned char>> &program_binaries)
{
    getDevice();

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    for (auto &prog_bin: program_binaries)
    {
        makeProgram(prog_bin.first, prog_bin.second);
    }
}

void Container::getDevice()
{
    // Get list of platforms
    cl::Platform::get(&platforms);
    if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");
    std::cout << "Found " << platforms.size() << " platforms\n";

    for(auto p : platforms)
    {
        std::cout << "Platform: " << p.getInfo<CL_PLATFORM_VENDOR>() << "\n";

        // Get list of GPU devices
        try
        {
            p.getDevices(DEVICE_TYPE, &devices);
        }
        catch (cl::Error &e) {}

        if (!devices.empty())
        {
            std::cout << "Found " << devices.size() << " devices\n";

            // Select device
            device = devices.at(0);
            platform = p;
            break;
        }
    }

    try
    {
        std::cout << "Using: " << platform.getInfo<CL_PLATFORM_VENDOR>() << ", " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    }
    catch (cl::Error &e)
    {
        throw std::runtime_error("No device found.\n");
    }
}

void Container::makeProgram(const std::string &program_name, const std::string &program_source)
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

void Container::makeProgram(const std::string &program_name, const std::vector<unsigned char> &program_bin)
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

Kernel Container::getKernel(const std::string &file_name, const std::string &kernel_name)
{
    return Kernel(*this, programs[file_name], kernel_name.c_str());
}

long Container::durationOfEvent(const cl::Event &event) const
{
    cl_ulong start;
    cl_ulong end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    return end - start;
}

}
