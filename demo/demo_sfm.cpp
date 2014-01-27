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
#include <stdint.h>

#include <viso_mono.h>
#include <reconstruction.h>
#include <png++/png.hpp>

using namespace std;

//#define N_FRAMES 373
#define N_FRAMES 20

int main (int argc, char** argv)
{
    if (argc<2)
    {
        cerr << "Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019" << endl;
        return 1;
    }
    string dir = argv[1];

    VisualOdometryMono::parameters param;
    param.calib.f  = 645.24; // focal length in pixels
    param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
    param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
    param.height = 1.6;
    param.pitch  = -0.08;
    param.bucket.max_features = 1000; // disable bucketing;

    // init visual odometry
    VisualOdometryMono viso(param);

    Reconstruction reconstruction;
    reconstruction.setCalibration(param.calib.f, param.calib.cu, param.calib.cv);

    // pose at each frame (this matrix transforms a point from the current
    // frame's camera coordinates to the first frame's camera coordinates)
    vector<Matrix> Tr_total;

    bool replace = false;

    for (int32_t i=0; i<N_FRAMES; i++)
    {
        char base_name[256]; sprintf(base_name,"%06d.png",i);
        string left_img_file_name  = dir + "/I1_" + base_name;

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

            reconstruction.update(viso.getMatches(),viso.getMotion(), 2, 2, 30, 3);

            replace = false;
        }
        else
        {
            Tr_total.push_back(Tr_total.back());
            cout << " ... failed!" << endl;
            replace = true;
        }
    }

    vector<Reconstruction::point3d> points = reconstruction.getPoints();
    for (std::vector<Reconstruction::point3d>::iterator point = points.begin(); point != points.end(); point++)
    {
        cout << point->x << ' '
             << point->y << ' '
             << point->z << endl;
    }



    cout << "Demo complete!" << endl;
    return 0;
}

