#include <viso_mono_cl.h>
#include "timer.hh"

using namespace std;

class CLInlierFinder
{
private:
    OpenCL::Container &cl_container;

    OpenCL::Kernel kernel_get_inlier;
    OpenCL::Kernel kernel_sum;
    OpenCL::Kernel kernel_update_inliers;

    const unsigned n_matches;
    const unsigned iters_per_batch;
    const size_t work_group_size;
    const size_t cl_groups_per_iter;
    const size_t cl_n_matches;

    typedef Matcher::p_match match_t;
    struct cl_match_t
    {
        cl_float u1p;
        cl_float v1p;
        cl_float u1c;
        cl_float v1c;
    };

    OpenCL::Buffer<cl_match_t> buff_matches;

    OpenCL::Buffer<cl_double> buff_fund_mat;
    OpenCL::Buffer<cl_uchar> buff_inlier_mask;
    OpenCL::Buffer<cl_ushort> buff_counts;
    OpenCL::Buffer<cl_uchar> buff_best_inlier_mask;
    OpenCL::Buffer<cl_ushort> buff_best_count;

    std::vector<cl::Event> update_deps;


    const std::vector<cl_match_t> matches;
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
    CLInlierFinder(const vector<Matcher::p_match> &p_matched, OpenCL::Container &cl_container, unsigned iters_per_batch, float inlier_threshold)
        :   cl_container (cl_container)
        ,   kernel_get_inlier (cl_container.getKernel("plane_and_inliers.cl", "find_inliers"))
        ,   kernel_sum (cl_container.getKernel("plane_and_inliers.cl", "sum"))
        ,   kernel_update_inliers (cl_container.getKernel("plane_and_inliers.cl", "update_best_inliers"))
        ,   n_matches (p_matched.size())
        ,   iters_per_batch (iters_per_batch)
        ,   work_group_size (kernel_get_inlier.local_size[0])
        ,   cl_groups_per_iter ((n_matches + work_group_size - 1)/work_group_size)
        ,   cl_n_matches (cl_groups_per_iter * work_group_size)
        ,   buff_matches (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_fund_mat (cl_container, CL_MEM_READ_ONLY, 9*iters_per_batch)
        ,   buff_inlier_mask (cl_container, CL_MEM_READ_WRITE, cl_n_matches*iters_per_batch)
        ,   buff_counts (cl_container, CL_MEM_READ_WRITE, iters_per_batch)
        ,   buff_best_inlier_mask (cl_container, CL_MEM_READ_WRITE, n_matches)
        ,   buff_best_count (cl_container, CL_MEM_READ_WRITE, cl_groups_per_iter)
        ,   matches (map<cl_match_t,match_t> (p_matched, [](const match_t &p) {return cl_match_t{p.u1p, p.v1p, p.u1c, p.v1c};}))
        ,   zeros (cl_groups_per_iter, 0)
    {
        update_deps.push_back( buff_matches.write(matches.data()) );
        update_deps.push_back( buff_best_count.write(zeros.data()) );

        kernel_get_inlier.setRange(cl::NDRange(iters_per_batch*cl_n_matches))
                .arg(cl_uint(n_matches))
                .arg(cl_uint(cl_n_matches))
                .arg(buff_matches)
                .arg(cl_float(inlier_threshold))
                .arg(buff_fund_mat)
                .arg(buff_inlier_mask)
                ;

        kernel_sum.setRange(cl::NDRange(iters_per_batch*work_group_size))
                .arg(buff_inlier_mask)
                .arg(buff_counts)
                .arg(cl_uint(n_matches))
                .arg(cl_uint(cl_n_matches))
                .arg(cl::__local(work_group_size*sizeof(cl_ushort)))
                ;

        kernel_update_inliers.setRange(cl::NDRange(cl_n_matches))
                .arg(buff_inlier_mask)
                .arg(buff_counts)
                .arg(cl_uint(iters_per_batch))
                .arg(cl_uint(n_matches))
                .arg(cl_uint(cl_n_matches))
                .arg(buff_best_inlier_mask)
                .arg(buff_best_count)
//                .arg(cl::__local(work_group_size*sizeof(cl_ushort)))
                ;
    }

    void update(const std::vector<Matrix> &F_estimates)
    {
        std::vector<double> F_array(9*iters_per_batch);
        for (unsigned i=0; i<iters_per_batch; i++)
        {
            for (unsigned j=0; j<9; j++)
            {
                F_array[i*9+j] = F_estimates[i].val[0][j];
            }
        }

        cl::Event write_f_event = buff_fund_mat.write(F_array.data(), update_deps);
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
#ifdef __arm__
    static const unsigned iters_per_batch = 2048;
#else
    static const unsigned iters_per_batch = 16;
#endif

    std::vector<Matrix> F_estimates(iters_per_batch);
    CLInlierFinder inlier_finder(p_matched, cl_container, iters_per_batch, param.inlier_threshold);

    for (int32_t k=0; k<param.ransac_iters; k+=iters_per_batch)
    {
        // initial RANSAC estimates of F
        for (auto &F : F_estimates)
        {
            // draw random sample set
            vector<int32_t> active = getRandomSample(p_matched.size(), 8);

            // estimate fundamental matrix and get inliers
            fundamentalMatrix(p_matched, active, F);
        }

        inlier_finder.update(F_estimates);
    }

    inliers = inlier_finder.getBestInliers();

    Matrix F;
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



class CLBestPlaneFinder
{
private:
    OpenCL::Container &cl_container;

    OpenCL::Kernel kernel_calc_dists;

    const vector<float> &d;

    const unsigned d_len;
    const size_t work_group_size;
    const size_t cl_sum_n_groups;
    const size_t cl_d_len;

    OpenCL::Buffer<cl_float> buff_d;
    OpenCL::Buffer<cl_float> buff_sums;

public:
    CLBestPlaneFinder(OpenCL::Container &cl_container, const vector<float> &d, float weight, float threshold)
        :   cl_container (cl_container)
        ,   kernel_calc_dists (cl_container.getKernel("plane_and_inliers.cl", "plane_calc_sums"))
        ,   d (d)
        ,   d_len (d.size())
        ,   work_group_size (kernel_calc_dists.local_size[0])
        ,   cl_sum_n_groups ((d_len + work_group_size - 1)/work_group_size)
        ,   cl_d_len (cl_sum_n_groups * work_group_size)
        ,   buff_d (cl_container, CL_MEM_READ_ONLY, d_len)
        ,   buff_sums (cl_container, CL_MEM_WRITE_ONLY, cl_d_len)
    {
        kernel_calc_dists.setRange(cl::NDRange(cl_d_len))
                .arg(buff_d)
                .arg(d_len)
                .arg(threshold)
                .arg(weight)
                .arg(buff_sums)
                ;
    }

    std::vector<float> get_dist_sums()
    {
        cl::Event write_event = buff_d.write(d.data());
        cl::Event calc_event = kernel_calc_dists.start({write_event});

        std::vector<float> sums (cl_d_len);
        cl::Event read_event = buff_sums.read_into(sums.data(), {calc_event});
        read_event.wait();

        std::cout << "write: " << cl_container.durationOfEvent(write_event) << "  ";
        std::cout << "calc: " << cl_container.durationOfEvent(calc_event) << "  ";
        std::cout << "read: " << unsigned(cl_container.durationOfEvent(read_event)) << "  ";
        std::cout << std::endl;

        return sums;
    }
};


double VisualOdometryMono_CL::findBestPlane(const Matrix &x_plane, double threshold, double weight)
{
    const double s_pitch = sin(-param.pitch);
    const double c_pitch = cos(-param.pitch);
    vector<float> d (x_plane.n);

    for (unsigned i=0; i<d.size(); i++) d[i] = c_pitch*x_plane.val[0][i];
    for (unsigned i=0; i<d.size(); i++) d[i] += s_pitch*x_plane.val[1][i];

    CLBestPlaneFinder plane_finder(cl_container, d, weight, threshold);
    const auto dist_sums = plane_finder.get_dist_sums();

    float   best_sum = 0;
    int32_t  best_idx = 0;

    for (unsigned i=0; i<d.size(); i++)
    {
        float sum = dist_sums[i];
        if (sum>best_sum)
        {
            best_sum = sum;
            best_idx = i;
        }
    }
    return d[best_idx];
}

