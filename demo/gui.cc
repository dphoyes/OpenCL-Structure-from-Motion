#include "gui.hh"

#ifndef __arm__

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#define GLM_FORCE_RADIANS
#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include "shader_srcs.generated.hh"

PointCloudViewer::PointCloudViewer()
{
    std::unique_lock<std::mutex> lock(running_mutex);
    run_thread = std::thread(&PointCloudViewer::run, this);
    running_cv.wait(lock, [&]{return running;});
}

void PointCloudViewer::run()
{
    sf::RenderWindow window{sf::VideoMode(640, 480), "Structure from Motion"};
    window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(30);

    if (glewInit() != GLEW_OK) throw std::runtime_error("Initialising GLEW failed.");
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

    GLuint program = makeShaders("vertex.glsl", "fragment.glsl");
    glUseProgram(program);
    view_uniform = glGetUniformLocation(program, "view");
    projection_uniform = glGetUniformLocation(program, "projection");
    colour_uniform = glGetUniformLocation(program, "colour");
    horiz_offset_uniform = glGetUniformLocation(program, "horiz_offset");

    point_cloud_obj = makeObject();
    axes_obj = makeObject();

    {
        std::array<glm::vec3,6> axes_geom = {
            glm::vec3{-2,0,0}, glm::vec3{2,0,0},
            glm::vec3{0,0,-0.00001}, glm::vec3{0,0,5},
            glm::vec3{0,0.00001,0}, glm::vec3{0,-2,0},
        };
        glBindBuffer(GL_ARRAY_BUFFER, axes_obj.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Reconstruction::point3d)*axes_geom.size(), &axes_geom[0], GL_STATIC_DRAW);
    }

    updateViewFromMouse(sf::Vector2i{}, 0);

    {
        std::lock_guard<std::mutex> lock(running_mutex);
        running = true;
        running_cv.notify_one();
    }

    while (running)
    {
        processEvents(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto render_sides = render_in_stereo ? std::vector<int>{-1,1} : std::vector<int>{0};
        sf::Vector2u win_size = window.getSize();

        for (int side : render_sides)
        {
            if (side)
            {
                glViewport(side<0 ? 0 : win_size.x/2, 0, win_size.x/2, win_size.y);
            }
            glUniform1f(horiz_offset_uniform, side * stereo_separation);

            glBindVertexArray(point_cloud_obj.vao);
            setColour(1,1,1);
            {
                std::lock_guard<std::mutex> lock(point_cloud_buffer_mutex);
                if (point_cloud_changed)
                {
                    glBindBuffer(GL_ARRAY_BUFFER, point_cloud_obj.vbo);
                    glBufferData(GL_ARRAY_BUFFER, sizeof(Reconstruction::point3d)*point_cloud->size(), &point_cloud->front(), GL_STATIC_DRAW);
                    point_cloud_changed = false;
                }
                if (point_cloud != nullptr)
                {
                    glDrawArrays(GL_POINTS, 0, point_cloud->size());
                }
            }

            glBindVertexArray(axes_obj.vao);
            setColour(0,0.8,0);
            glDrawArrays(GL_LINES, 0, 6);
        }

        if (render_sides.size() == 2)
        {
            glUseProgram(0);
            glViewport(0, 0, win_size.x, win_size.y);
            glBegin(GL_TRIANGLES);
            glColor3b(0, 0, 0);
            glVertex3f(-0.01f, -1, 0);
            glVertex3f(0.01f, -1, 0);
            glVertex3f(-0.01f, 1, 0);
            glVertex3f(-0.01f, 1, 0);
            glVertex3f(0.01f, 1, 0);
            glVertex3f(0.01f, -1, 0);
            glEnd();
            glUseProgram(program);
        }


        window.display();
    }
}

void PointCloudViewer::update(const std::vector<Reconstruction::point3d> &points)
{
    std::lock_guard<std::mutex> lock(point_cloud_buffer_mutex);
    point_cloud = &points;
    point_cloud_changed = true;
}

void PointCloudViewer::waitClose()
{
    run_thread.join();
}

void PointCloudViewer::processEvents(sf::RenderWindow &window)
{
    bool window_size_changed = false;
    static bool button_pressed = false;
    static sf::Vector2i prev_mouse_pos;
    sf::Vector2i total_mouse_drag;
    int wheel_delta = 0;

    sf::Event event;
    while (window.pollEvent(event))
    {
        switch (event.type)
        {
            case sf::Event::Closed:
            running = false;
            break;
            case sf::Event::Resized:
            window_size_changed = true;
            break;
            case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Middle || event.mouseButton.button == sf::Mouse::Left)
            {
                button_pressed = true;
                horiz_drag_dir = -up_dir;
            }
            break;
            case sf::Event::MouseButtonReleased:
            if (event.mouseButton.button == sf::Mouse::Middle || event.mouseButton.button == sf::Mouse::Left)
            {
                button_pressed = false;
            }
            break;
            case sf::Event::MouseMoved:
            {
               sf::Vector2i new_pos{event.mouseMove.x, event.mouseMove.y};
               if (button_pressed) total_mouse_drag += new_pos - prev_mouse_pos;
               prev_mouse_pos = new_pos;
           }
           break;
           case sf::Event::MouseWheelMoved:
           wheel_delta += event.mouseWheel.delta;
           break;
           case sf::Event::KeyPressed:
           if (event.key.code == sf::Keyboard::Key::S)
           {
            render_in_stereo = !render_in_stereo;
            updateProjectionFromWindow(window);
        }
        if (event.key.code == sf::Keyboard::Key::E) stereo_separation += 0.1;
        if (event.key.code == sf::Keyboard::Key::D) stereo_separation -= 0.1;
        break;
        default:
        break;
    }
}

if (window_size_changed) updateProjectionFromWindow(window);
if (total_mouse_drag.x || total_mouse_drag.y || wheel_delta) updateViewFromMouse(total_mouse_drag, wheel_delta);
}

void PointCloudViewer::updateViewFromMouse(const sf::Vector2i &drag, int wheel_delta)
{
    const float speed = 0.01;
    static int view_zoom = -15;
    static glm::vec2 view_angle {3.37, -0.21};

    view_zoom += wheel_delta;
    float view_radius = 5 * glm::pow(1.1f, float(-view_zoom)); // The forward/backward position is a function of the scroll wheel position

    stereo_separation = 0.1 + view_radius * view_radius / 600;

    view_angle.x += horiz_drag_dir * speed * drag.x;
    view_angle.x = glm::mod(view_angle.x, 2 * glm::pi<float>()); // Wrap the angle from 0 to 360 degrees
    view_angle.y -= speed * drag.y;
    view_angle.y = glm::mod(view_angle.y, 2 * glm::pi<float>()); // Wrap the angle from 0 to 360 degrees

    glm::vec3 view_centre{0,0,0};
    glm::vec3 cam_position = view_centre;
    cam_position.x += view_radius * glm::sin(view_angle.x) * glm::cos(view_angle.y);
    cam_position.y += view_radius * glm::sin(view_angle.y);
    cam_position.z += view_radius * glm::cos(view_angle.x) * glm::cos(view_angle.y);

    up_dir = view_angle.y < glm::half_pi<float>() || view_angle.y > 3 * glm::half_pi<float>() ? -1 : 1;

    glm::mat4 view = glm::lookAt(
        cam_position,
        view_centre,
        glm::vec3(0,up_dir,0)
        );
    glUniformMatrix4fv(view_uniform, 1, GL_FALSE, &view[0][0]);
}

void PointCloudViewer::setColour(float r, float g, float b)
{
    glm::vec3 colour{r,g,b};
    glUniform3fv(colour_uniform, 1, &colour.x);
}

PointCloudViewer::obj_t PointCloudViewer::makeObject()
{
    obj_t ret;
    glGenVertexArrays(1, &ret.vao);
    glBindVertexArray(ret.vao);

    glGenBuffers(1, &ret.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, ret.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    return ret;
}

void PointCloudViewer::updateProjectionFromWindow(const sf::RenderWindow &window)
{
    sf::Vector2u size = window.getSize();
    unsigned width = render_in_stereo? size.x/2 : size.x;
    glViewport(0, 0, width, size.y);
    glm::mat4 projection = glm::perspective(0.785f, float(width) / float(size.y), 0.00001f, 1000.0f);
    glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, &projection[0][0]);
}

GLuint PointCloudViewer::makeShader(const std::string &src_name, const GLenum shader_type) const
{
    const std::string &src = SHADER_SRCS.at(src_name);
    GLuint shader_id = glCreateShader(shader_type);

    // compile
    const GLchar *src_array = src.c_str();
    glShaderSource(shader_id, 1, &src_array, NULL);
    glCompileShader(shader_id);

    // Die if errors occurred
    GLint success;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        // Print any errors
        GLint infoLength;
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &infoLength);
        GLchar info[infoLength];
        glGetShaderInfoLog(shader_id, infoLength, NULL, info);
        std::cerr << info;
        throw std::runtime_error("Compiling shader failed: " + std::string(src_name));
    }

    return shader_id;
}

GLuint PointCloudViewer::makeShaders(const std::string &vertex_src, const std::string &fragment_src) const
{
    // Compile
    std::array<GLuint, 2> all_shaders;
    all_shaders[0] = makeShader(vertex_src, GL_VERTEX_SHADER);
    all_shaders[1] = makeShader(fragment_src, GL_FRAGMENT_SHADER);

    // Link
    GLuint program = glCreateProgram();
    for (GLuint shader : all_shaders) glAttachShader(program, shader);
        glLinkProgram(program);

    // Die if errors occurred
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        // Print any errors
        GLint infoLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLength);
        GLchar info[infoLength];
        glGetProgramInfoLog(program, infoLength, NULL, info);
        std::cerr << info;
        throw std::runtime_error("Linking shaders failed.");
    }

    // Now that program is compiled, delete the original shaders
    for (GLuint shader : all_shaders) glDeleteShader(shader);
        return program;
}

#endif
