#ifndef OPENCLCONTAINER_H
#define OPENCLCONTAINER_H

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <fstream>
#include <string>
#include <unordered_map>

namespace OpenCL
{

class Task;
class Kernel;
class Container;
template <typename T> class Buffer;

class Container
{
public:
    cl::Context context;

    std::vector<cl::Platform> platforms;
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;

    std::array<cl::CommandQueue, 2> queues;
    std::unordered_map<std::string,cl::Program> programs;

    const cl_device_type DEVICE_TYPE = 0;
    const unsigned N_WORK_GROUPS = 0;
    const unsigned WORK_GROUP_SIZE = 0;
    const unsigned N_WORK_ITEMS = 0;

    Container(cl_device_type device_type, unsigned n_work_groups, unsigned work_group_size);
    Container(const Container &c);

    template<typename T>
    void init(const T &program_srcs)
    {
        getDevice();
        context = cl::Context(devices);

        for (auto &q : queues)
        {
            q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }

        for (auto &prog_src: program_srcs)
        {
            makeProgram(prog_src.first, prog_src.second);
        }
    }

    void getDevice();
    void makeProgram(const std::string &program_name, const std::string &program_source);
    void makeProgram(const std::string &program_name, const std::vector<unsigned char> &program_bin);
    Kernel getKernel(const std::string &file_name, const std::string &kernel_name);
    long durationOfEvent(const cl::Event &event) const;
};


class GPUContainer: public Container
{
public:
    GPUContainer();
};
class CPUContainer: public Container
{
public:
    CPUContainer();
};
class FPGAContainer: public Container
{
public:
    FPGAContainer();
};


template <typename T>
class Buffer
{
public:
    Container &cl_container;
    size_t size;
    cl::Buffer buff;
    unsigned queue_id = 0;

    Buffer(Container &cl_container, cl_mem_flags flags, unsigned N)
        :   cl_container (cl_container)
        ,   size(N*sizeof(T))
        ,   buff(cl_container.context, flags, size)
    {}

    Buffer& setQueue(const unsigned id)
    {
        queue_id = id;
        return *this;
    }

    cl::Event write(const void *data, const std::vector<cl::Event> &deps = {})
    {
        cl::Event ev;
        cl_container.queues[queue_id].enqueueWriteBuffer(buff, CL_FALSE, 0, size, data, &deps, &ev);
        return ev;
    }

    cl::Event read_into(void *data, const std::vector<cl::Event> &deps = {})
    {
        cl::Event ev;
        cl_container.queues[queue_id].enqueueReadBuffer(buff, CL_FALSE, 0, size, data, &deps, &ev);
        return ev;
    }
};


class Kernel
{
public:
    Container &cl_container;
    cl::Kernel kernel;

    cl::size_t<3> reqd_local_size;

    cl::NDRange offset;
    cl::NDRange global_size;
    cl::NDRange local_size;

    unsigned queue_id = 0;
    unsigned current_arg_num = 0;

    Kernel(Container &cl_container, const cl::Program& program, const char* name);

    virtual cl::Event start(const std::vector<cl::Event> &deps = {});

    template<typename T>
    __attribute__((always_inline)) Kernel& arg(T val)
    {
        kernel.setArg(current_arg_num, val);
        current_arg_num++;
        return *this;
    }

    template<typename T>
    __attribute__((always_inline)) Kernel& arg(unsigned id, T val)
    {
        current_arg_num = id;
        return arg(val);
    }

    template<typename T> __attribute__((always_inline)) Kernel& arg(Buffer<T> &val)               {return arg(val.buff);}
    template<typename T> __attribute__((always_inline)) Kernel& arg(unsigned id, Buffer<T> &val)  {return arg(id, val.buff);}

    Kernel& setRange(const cl::NDRange &global);
    Kernel& setQueue(const unsigned id);
};


class Task : public Kernel
{
public:
    Task(const Kernel &k): Kernel(k) {}
    cl::Event start(const std::vector<cl::Event> &deps = {}) override;
};



} // namespace

#endif // OPENCLCONTAINER_H
