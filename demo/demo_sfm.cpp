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
#include <iomanip>
#include <chrono>
#include <stdint.h>
#include <unistd.h>

#include <viso_mono.h>
#include <reconstruction.h>
#include <png++/png.hpp>
#include <ply_exporter.h>

using namespace std;

//#define N_FRAMES 373
//#define N_FRAMES 50
#define N_FRAMES 41

int main (int argc, char** argv)
{
    VisualOdometryMono::parameters param;

    int optFlag;
    while((optFlag = getopt (argc, argv, "f:u:v:h:p:")) != -1)
    {
        switch(optFlag)
        {
            case 'f':
                param.calib.f = atof(optarg); // focal length in pixels
                break;
            case 'u':
                param.calib.cu = atof(optarg); // principal point (u-coordinate) in pixels
                break;
            case 'v':
                param.calib.cv = atof(optarg); // principal point (v-coordinate) in pixels
                break;
            case 'h':
                param.height = atof(optarg); // camera height above ground (meters)
                break;
            case 'p':
                param.pitch = atof(optarg); // camera pitch (rad, negative=pointing down)
                break;
            default:
                abort ();
        }
    }
    param.bucket.max_features = 1000; // disable bucketing;

    if (optind >= argc)
    {
        cerr << "Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019" << endl;
        exit(1);
    }
    string dir = argv[optind];

    cout << "Directory: " << dir << "\n"
         << "f: " << param.calib.f << " pixels \n"
         << "cu: " << param.calib.cu << " pixels \n"
         << "cv: " << param.calib.cv << " pixels \n"
         << "height: " << param.height << " m\n"
         << "pitch: " << param.pitch << endl;

    // init visual odometry
    VisualOdometryMono viso(param);

    Reconstruction reconstruction;
    reconstruction.setCalibration(param.calib.f, param.calib.cu, param.calib.cv);

    // pose at each frame (this matrix transforms a point from the current
    // frame's camera coordinates to the first frame's camera coordinates)
    vector<Matrix> Tr_total;

    bool replace = false;

    auto t0 = chrono::high_resolution_clock::now();

    for (int32_t i=0; i<N_FRAMES; i++)
    {
//        char base_name[256]; sprintf(base_name,"%06d.png",i);
//        string left_img_file_name  = dir + "/I1_" + base_name;
        char base_name[256]; sprintf(base_name,"%04d.png",i);
        string left_img_file_name  = dir + "/" + base_name;

        png::image<png::gray_pixel> left_img(left_img_file_name);
        int32_t width  = left_img.get_width();
        int32_t height = left_img.get_height();

        // convert input images to uint8_t buffer
        uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
        int32_t k=0;
        for (int32_t v=0; v<height; v++)
        {
            for (int32_t u=0; u<width; u++)
            {
                left_img_data[k]  = left_img.get_pixel(u,v);
                k++;
            }
        }

        // compute visual odometry
        cout << "Processing: Frame: " << i;
        int32_t dims[] = {width,height,width};
        bool viso_success = viso.process(left_img_data,dims,replace);
        free(left_img_data);

        if (i==0)
        {
            Tr_total.push_back(Matrix::eye(4));
            cout << endl;
        }
        else if (viso_success)
        {
            Matrix motion = Matrix::inv(viso.getMotion());
            Tr_total.push_back( Tr_total.back() * motion );

            // print stats
            double num_matches = viso.getNumberOfMatches();
            double num_inliers = viso.getNumberOfInliers();
            cout << ", Matches: " << num_matches;
            cout << ", Inliers: " << 100.0*num_inliers/num_matches << "%" << ", Current pose: " << endl;
            cout << Tr_total.back() << endl << endl;

//            reconstruction.update(viso.getMatches(),viso.getMotion(), 2, 2, 30, 3);
            reconstruction.update(viso.getMatches(),viso.getMotion(), 0, 2, 30, 3);

            replace = false;
        }
        else
        {
            Tr_total.push_back(Tr_total.back());
            cout << " ... failed!" << endl;
            replace = true;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();

    auto execution_time = chrono::duration_cast<chrono::microseconds>(t1 - t0);
    double time_seconds = (execution_time.count()*1e-6);
    double fps = N_FRAMES / time_seconds;
    cout << "Total time: " << time_seconds << " s" << endl;
    cout << "FPS: " << fps << endl;

//    reconstruction.update(std::vector<Matcher::p_match>(),Matrix::eye(4), 0, 2, 60, 3);

    export_ply(reconstruction.getPoints(), dir + "/viso_cloud.ply");

    cout << "Demo complete!" << endl;
    return 0;
}

