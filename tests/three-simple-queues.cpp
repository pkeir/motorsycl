#include <sycl/sycl.hpp>

int main(int argc, char *argv[])
{
  using namespace sycl;

  queue q1;
  queue q2;
  queue q3;

  device d1 = q1.get_device();
  device d2 = q2.get_device();
  device d3 = q3.get_device();

  context c1 = q1.get_context();
  context c2 = q2.get_context();
  context c3 = q3.get_context();

  platform p1 = c1.get_platform();
  platform p2 = c2.get_platform();
  platform p3 = c3.get_platform();

  backend b1 = p1.get_backend();
  backend b2 = p2.get_backend();
  backend b3 = p3.get_backend();

  assert((d1==d2) && (d2==d3));
  assert((c1==c2) && (c2==c3));
  assert((p1==p2) && (p2==p3));
  assert((b1==b2) && (b2==b3));

  return 0;
}
