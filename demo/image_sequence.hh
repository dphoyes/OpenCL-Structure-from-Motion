#include <png++/png.hpp>

class ImageSequenceLoader
{
    const std::string dir;
    uint32_t width;
    uint32_t height;
    std::array<uint32_t, 3> dims;
    std::vector<uint8_t> img_data;

public:

    ImageSequenceLoader(std::string dir): dir(dir)
    {
        auto img = this->loadPng(0);
        width = img.get_width();
        height = img.get_height();
        dims = {width, height, width};
        img_data.resize(width*height);
    }

    png::image<png::gray_pixel> loadPng(uint32_t i)
    {
        char base_name[256];
        sprintf(base_name, "%04d.png", i);
        std::string img_file_path = dir + "/" + base_name;

        png::image<png::gray_pixel> img(img_file_path);
        return img;
    }

    uint8_t* getFrame(uint32_t i)
    {
        auto img = loadPng(i);
        if (width != img.get_width() || height != img.get_height())
        {
            throw std::runtime_error("Image dimension inconsistency");
        }

        // convert input images to uint8_t buffer
        int32_t k=0;
        for (uint32_t v=0; v<height; v++)
        {
            for (uint32_t u=0; u<width; u++)
            {
                img_data[k] = img.get_pixel(u,v);
                k++;
            }
        }
        return &img_data[0];
    }

    std::array<uint32_t, 3> getDims()
    {
        return dims;
    }
};
