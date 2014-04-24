/*
Copyright 2012. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <iomanip>
#include <chrono>
#include <stdint.h>
#include <unistd.h>
#include <stdexcept>

#include "ply_exporter.hh"
#include "gui.hh"
#include "sfm.hh"
#include "image_sequence.hh"

using namespace std;

struct cmd_params_t
{
    VisualOdometryMono::parameters viso;
    string dir;
    unsigned n_frames = 0;
    bool use_gui = false;
    bool use_opencl = false;
};

cmd_params_t getParams(int argc, char** argv)
{
    cmd_params_t param;


    int optFlag;
    while((optFlag = getopt (argc, argv, "f:u:v:h:p:gcn:")) != -1)
    {
        float fval = optarg != nullptr ? atof(optarg) : 0;
        float ival = optarg != nullptr ? atoi(optarg) : 0;
        switch(optFlag)
        {
        case 'f':
            param.viso.calib.f = fval; // focal length in pixels
            break;
        case 'u':
            param.viso.calib.cu = fval; // principal point (u-coordinate) in pixels
            break;
        case 'v':
            param.viso.calib.cv = fval; // principal point (v-coordinate) in pixels
            break;
        case 'h':
            param.viso.height = fval; // camera height above ground (meters)
            break;
        case 'p':
            param.viso.pitch = fval; // camera pitch (rad, negative=pointing down)
            break;
        case 'g':
            param.use_gui = true; // enable gui
            break;
        case 'c':
            param.use_opencl = true; // enable OpenCL
            break;
        case 'n':
            param.n_frames = ival; // number of frames to process
            break;
        default:
            abort ();
        }
    }
    param.viso.bucket.max_features = 1000; // disable bucketing;

    if (optind >= argc)
    {
        cerr << "Specify image directory" << endl;
        exit(1);
    }
    param.dir = argv[optind];

    cout << "Directory: " << param.dir << "\n"
         << "n frames: " << param.n_frames << "\n"
         << "use gui: " << (param.use_gui ? "yes" : "no") << "\n"
         << "f: " << param.viso.calib.f << " pixels \n"
         << "cu: " << param.viso.calib.cu << " pixels \n"
         << "cv: " << param.viso.calib.cv << " pixels \n"
         << "height: " << param.viso.height << " m\n"
         << "pitch: " << param.viso.pitch << endl;

    return param;
}


int main (int argc, char** argv)
{
    auto param = getParams(argc, argv);

    ImageSequenceLoader video(param.dir);
    StructureFromMotion sfm(param.viso, video.getDims(), param.use_opencl);

    std::unique_ptr<PointCloudViewerInterface> gui;
    if (param.use_gui) gui.reset(new PointCloudViewer);
    else gui.reset(new NoPointCloudViewer);

    auto t0 = chrono::high_resolution_clock::now();

    for (int32_t i=0; i<param.n_frames; i++)
    {
        uint8_t* img_data = video.getFrame(i);
        cout << "Processing: Frame: " << i;
        sfm.update(img_data);
        gui->update(sfm.getPoints());
    }

    auto t1 = chrono::high_resolution_clock::now();

    auto execution_time = chrono::duration_cast<chrono::microseconds>(t1 - t0);
    double time_seconds = (execution_time.count()*1e-6);
    double fps = param.n_frames / time_seconds;
    cout << "Total time: " << time_seconds << " s" << endl;
    cout << "FPS: " << fps << endl;

    export_ply(sfm.getPoints(), param.dir + "/viso_cloud.ply");

    cout << "Demo complete!" << endl;
    gui->waitClose();
    return 0;
}

