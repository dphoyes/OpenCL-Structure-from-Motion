#ifndef POINT3D_HH
#define POINT3D_HH

// a generic 3d point
struct Point3d {
  float x,y,z;
  Point3d () {}
  Point3d (float x,float y,float z) : x(x),y(y),z(z) {}
};

#endif // POINT3D_HH
