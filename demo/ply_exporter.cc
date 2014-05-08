#include "ply_exporter.hh"

int export_ply(const std::vector<Point3d> &points, const std::string &filename)
{
    std::ofstream out(filename);

    if (!out.is_open())
    {
        std::cerr << "File " << filename << " could not be opened" << std::endl;
        return 1;
    }

    /* Start outputting PLY File */

    out << "ply\n"
        << "format ascii 1.0\n"
        << "comment Created by Libviso2 PLY Formatter\n"
        << "element vertex " << points.size() << '\n'
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "end_header\n";

    for (const auto &p : points)
    {
        out << p.x << ' '
            << p.y << ' '
            << p.z << '\n';
    }

    /* End of PLY File */
    out.close();
    return 0;
}
