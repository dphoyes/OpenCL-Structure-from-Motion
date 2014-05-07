#ifndef GUI_H__
#define GUI_H__

#include <reconstruction.h>

class PointCloudViewerInterface
{
public:
    virtual void run() = 0;
    virtual void update(const std::vector<Reconstruction::point3d> &points) = 0;
    virtual void waitClose() = 0;
};

class NoPointCloudViewer: public PointCloudViewerInterface
{
public:
    void run() override {}
    void update(const std::vector<Reconstruction::point3d> &points) override {}
    void waitClose() override {}
};

#ifdef __arm__
typedef NoPointCloudViewer PointCloudViewer;
#else

#include <thread>
#include <mutex>
#include <condition_variable>
#include <GL/glew.h>
#include <SFML/Graphics.hpp>

class PointCloudViewer: public PointCloudViewerInterface
{
    struct obj_t
    {
        GLuint vao;
        GLuint vbo;
    };

    bool render_in_stereo = false;
    float stereo_separation = 1;

    signed char up_dir = -1;
    signed char horiz_drag_dir = 1;

    GLuint view_uniform;
    GLuint projection_uniform;
    GLuint colour_uniform;
    GLuint horiz_offset_uniform;

    obj_t point_cloud_obj;
    obj_t axes_obj;

    const std::vector<Reconstruction::point3d> *point_cloud = nullptr;
    bool point_cloud_changed = false;
    std::mutex point_cloud_buffer_mutex;

    std::thread run_thread;

    std::condition_variable cv;

    bool running = false;
    std::mutex running_mutex;
    std::condition_variable running_cv;


public:
    PointCloudViewer();
    void run() override;
    void update(const std::vector<Reconstruction::point3d> &points) override;
    void waitClose() override;

private:
    void processEvents(sf::RenderWindow &window);
    void updateViewFromMouse(const sf::Vector2i &drag, int wheel_delta);
    void setColour(float r, float g, float b);
    obj_t makeObject();
    void updateProjectionFromWindow(const sf::RenderWindow &window);

    GLuint makeShader(const std::string &filename, const GLenum shader_type) const;
    GLuint makeShaders(const std::string &vertex_src, const std::string &fragment_src) const;
};

#endif

 
#endif /* end of include guard: GUI_H__ */
