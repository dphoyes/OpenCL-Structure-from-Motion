#ifndef PLY_EXPORTER_H
#define PLY_EXPORTER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <reconstruction.h>

int export_ply(const std::vector<Reconstruction::point3d> &points, const std::string &filename);

#endif // PLY_EXPORTER_H
