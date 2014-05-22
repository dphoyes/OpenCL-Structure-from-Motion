#include <viso_mono_cl.h>

using namespace std;

class CLInlierFinder
{
private:
    OpenCL::Container &cl_container;

    OpenCL::Kernel kernel_get_inlier;
    OpenCL::Kernel kernel_sum;
    OpenCL::Kernel kernel_update_inliers;

    const unsigned n_matches;
    const size_t work_group_size = 128;
    const size_t n_match_groups;
    const size_t cl_n_matches;
    const unsigned n_counts = 2;

    OpenCL::Buffer<cl_float> buff_match_u1p;
    OpenCL::Buffer<cl_float> buff_match_v1p;
    OpenCL::Buffer<cl_float> buff_match_u1c;
    OpenCL::Buffer<cl_float> buff_match_v1c;

    OpenCL::Buffer<cl_double> buff_fund_mat;
    OpenCL::Buffer<cl_uchar> buff_inlier_mask;
    OpenCL::Buffer<cl_ushort> buff_counts;
    OpenCL::Buffer<cl_uchar> buff_best_inlier_mask;
    OpenCL::Buffer<cl_ushort> buff_best_count;

    std::vector<cl::Event> update_deps;

    typedef Matcher::p_match match_t;
    const std::vector<cl_float> match_u1p;
    const std::vector<cl_float> match_v1p;
    const std::vector<cl_float> match_u1c;
    const std::vector<cl_float> match_v1c;
    const std::vector<cl_ushort> zeros;

    template <typename oT, typename iT>
    std::vector<oT> map(const std::vector<iT> &i_vec, std::function<oT(iT)> func)
    {
        std::vector<oT> o_vec(i_vec.size());
        for (unsigned i=0; i<i_vec.size(); i++)
        {
            o_vec[i] = func(i_vec[i]);
        }
        return o_vec;
    }

public:
    CLInlierFinder(const vector<Matcher::p_match> &p_matched, OpenCL::Container &cl_container, float inlier_threshold)
        :   cl_container (cl_container)
        ,   kernel_get_inlier (cl_container.getKernel("inlier.cl", "find_inliers"))
        ,   kernel_sum (cl_container.getKernel("inlier.cl", "sum"))
        ,   kernel_update_inliers (cl_container.getKernel("inlier.cl", "update_best_inliers"))
        ,   n_matches (p_matched.size())
        ,   n_match_groups ((n_matches + work_group_size - 1)/work_group_size)
        ,   cl_n_matches (n_match_groups * work_group_size)
        ,   buff_match_u1p (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_match_v1p (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_match_u1c (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_match_v1c (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_fund_mat (cl_container, CL_MEM_READ_ONLY, 9)
        ,   buff_inlier_mask (cl_container, CL_MEM_READ_WRITE, n_matches)
        ,   buff_counts (cl_container, CL_MEM_READ_WRITE, n_counts)
        ,   buff_best_inlier_mask (cl_container, CL_MEM_READ_WRITE, n_matches)
        ,   buff_best_count (cl_container, CL_MEM_READ_WRITE, n_match_groups)
        ,   match_u1p (map<cl_float,match_t> (p_matched, [](const match_t &p) {return p.u1p;}))
        ,   match_v1p (map<cl_float,match_t> (p_matched, [](const match_t &p) {return p.v1p;}))
        ,   match_u1c (map<cl_float,match_t> (p_matched, [](const match_t &p) {return p.u1c;}))
        ,   match_v1c (map<cl_float,match_t> (p_matched, [](const match_t &p) {return p.v1c;}))
        ,   zeros (n_match_groups, 0)
    {
        update_deps.push_back( buff_match_u1p.write(match_u1p.data()) );
        update_deps.push_back( buff_match_v1p.write(match_v1p.data()) );
        update_deps.push_back( buff_match_u1c.write(match_u1c.data()) );
        update_deps.push_back( buff_match_v1c.write(match_v1c.data()) );
        update_deps.push_back( buff_best_count.write(zeros.data()) );

        kernel_get_inlier.setRanges(cl_n_matches, work_group_size)
                .arg(buff_match_u1p)
                .arg(buff_match_v1p)
                .arg(buff_match_u1c)
                .arg(buff_match_v1c)
                .arg(cl_uint(n_matches))
                .arg(cl_float(inlier_threshold))
                .arg(buff_fund_mat)
                .arg(buff_inlier_mask)
                ;

        kernel_sum.setRanges(n_counts*work_group_size, work_group_size)
                .arg(buff_inlier_mask)
                .arg(buff_counts)
                .arg(cl_uint(n_matches))
                .arg(cl::__local(work_group_size*sizeof(cl_ushort)))
                ;

        kernel_update_inliers.setRanges(cl_n_matches, work_group_size)
                .arg(buff_inlier_mask)
                .arg(buff_counts)
                .arg(cl_uint(n_counts))
                .arg(cl_uint(n_matches))
                .arg(buff_best_inlier_mask)
                .arg(buff_best_count)
                .arg(cl::__local(work_group_size*sizeof(cl_ushort)))
                ;
    }

    void update(Matrix &F)
    {
        cl::Event write_f_event = buff_fund_mat.write(&F.val[0][0], update_deps);

        cl::Event get_inlier_complete_event = kernel_get_inlier.start({write_f_event});
        cl::Event sum_complete_event = kernel_sum.start({get_inlier_complete_event});
        cl::Event update_complete_event = kernel_update_inliers.start({sum_complete_event});

        update_deps = {update_complete_event};
        write_f_event.wait();

//        cl::WaitForEvents(update_deps);
//        std::cerr << "write f: " << cl_container.durationOfEvent(write_f_event) << "  ";
//        std::cerr << "get_inlier: " << cl_container.durationOfEvent(get_inlier_complete_event) << "  ";
//        std::cerr << "sum: " << cl_container.durationOfEvent(sum_complete_event) << "  ";
//        std::cerr << "update: " << cl_container.durationOfEvent(update_complete_event) << "  ";
//        std::cerr << std::endl;
    }

    std::vector<int32_t> getBestInliers()
    {
        std::vector<uint8_t> inlier_mask (n_matches);
        buff_best_inlier_mask.read_into(inlier_mask.data(), update_deps).wait();

        std::vector<int32_t> inliers;
        // for all matches do
        for (unsigned i=0; i<inlier_mask.size(); i++)
        {
            if (inlier_mask[i])
            {
                inliers.push_back(i);
            }
        }
        return inliers;
    }

};

Matrix VisualOdometryMono_CL::ransacEstimateF(const vector<Matcher::p_match> &p_matched)
{
    CLInlierFinder inlier_finder(p_matched, cl_container, param.inlier_threshold);

    // initial RANSAC estimate of F
    Matrix F;
    for (int32_t k=0; k<param.ransac_iters; k++)
    {
        // draw random sample set
        vector<int32_t> active = getRandomSample(p_matched.size(),8);

        // estimate fundamental matrix and get inliers
        fundamentalMatrix(p_matched,active,F);
        inlier_finder.update(F);
    }

    inliers = inlier_finder.getBestInliers();

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


