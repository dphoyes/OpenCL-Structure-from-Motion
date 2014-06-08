#include <viso_mono_cl.h>
#include "timer.hh"

using namespace std;

class CLInlierFinder
{
private:
    OpenCL::Container &cl_container;

    OpenCL::Task kernel_get_inlier;

    const unsigned n_matches;
    const unsigned iters_per_batch;

    typedef Matcher::p_match match_t;
    struct cl_match_t
    {
        cl_float u1p;
        cl_float v1p;
        cl_float u1c;
        cl_float v1c;
    };

    OpenCL::Buffer<cl_match_t> buff_matches;
    OpenCL::Buffer<cl_float> buff_fund_mat;
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
        ,   n_matches (p_matched.size())
        ,   iters_per_batch (iters_per_batch)
        ,   buff_matches (cl_container, CL_MEM_READ_ONLY, n_matches)
        ,   buff_fund_mat (cl_container, CL_MEM_READ_ONLY, 9*iters_per_batch)
        ,   buff_best_inlier_mask (cl_container, CL_MEM_WRITE_ONLY, n_matches)
        ,   buff_best_count (cl_container, CL_MEM_READ_WRITE, 1)
        ,   matches (map<cl_match_t,match_t> (p_matched, [](const match_t &p) {return cl_match_t{p.u1p, p.v1p, p.u1c, p.v1c};}))
        ,   zeros (1, 0)
    {
        update_deps.push_back( buff_matches.write(matches.data()) );
        update_deps.push_back( buff_best_count.write(zeros.data()) );

        kernel_get_inlier
                .arg(cl_uint(n_matches))
                .arg(cl_uint(iters_per_batch))
                .arg(buff_matches)
                .arg(buff_fund_mat)
                .arg(cl_float(inlier_threshold))
                .arg(buff_best_inlier_mask)
                .arg(buff_best_count)
                ;
    }

    void update(const std::vector<Matrix> &F_estimates)
    {
        std::vector<float> F_array(9*iters_per_batch);
        for (unsigned i=0; i<iters_per_batch; i++)
        {
            for (unsigned j=0; j<9; j++)
            {
                F_array[i*9+j] = F_estimates[i].val[0][j];
            }
        }

        cl::Event write_f_event = buff_fund_mat.write(F_array.data(), update_deps);
        cl::Event get_inlier_complete_event = kernel_get_inlier.start({write_f_event});

        update_deps = {get_inlier_complete_event};
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

Matrix VisualOdometryMono_CL::ransacEstimateF_disabled(const vector<Matcher::p_match> &p_matched)
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

    OpenCL::Task kernel_calc;

    const unsigned d_len;

    OpenCL::Buffer<cl_double> buff_d;
    OpenCL::Buffer<cl_uint> buff_best;

public:
    CLBestPlaneFinder(OpenCL::Container &cl_container, unsigned d_len, double weight, double threshold)
        :   cl_container (cl_container)
        ,   kernel_calc (cl_container.getKernel("plane_and_inliers.cl", "plane_calc_sums"))
        ,   d_len (d_len)
        ,   buff_d (cl_container, CL_MEM_READ_ONLY, d_len)
        ,   buff_best (cl_container, CL_MEM_WRITE_ONLY, 1)
    {
        kernel_calc
                .arg(buff_d)
                .arg(d_len)
                .arg(threshold)
                .arg(weight)
                .arg(buff_best)
                ;
    }

    unsigned get_best(const vector<double> &d)
    {
        cl::Event write_event = buff_d.write(d.data());
        cl::Event calc_event = kernel_calc.start();

        unsigned best_idx;
        cl::Event read_event = buff_best.read_into(&best_idx);
        read_event.wait();

        std::cout << "write: " << cl_container.durationOfEvent(write_event) << "  ";
        std::cout << "calc: " << cl_container.durationOfEvent(calc_event) << "  ";
        std::cout << "read: " << unsigned(cl_container.durationOfEvent(read_event)) << "  ";
        std::cout << std::endl;

        return best_idx;
    }
};


double VisualOdometryMono_CL::findBestPlane(const Matrix &x_plane, double threshold, double weight)
{
    CLBestPlaneFinder plane_finder(cl_container, x_plane.n, weight, threshold);

    vector<double> d (x_plane.n);

    const double s_pitch = sin(-param.pitch);
    const double c_pitch = cos(-param.pitch);
    for (unsigned i=0; i<d.size(); i++) d[i] = c_pitch*x_plane.val[0][i];
    for (unsigned i=0; i<d.size(); i++) d[i] += s_pitch*x_plane.val[1][i];

    const unsigned best_idx = plane_finder.get_best(d);
    return d[best_idx];
}

