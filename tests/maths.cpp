#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>
#include <cmath>
#include <cassert>
#include <iostream>

class Foo;

template <typename T>
bool test_maths(const T tol)
{
  const float x{49.0f}, x_sqrt{7.0f};
  const auto near = [&](const auto& v, const auto &expected) {
    return v > (expected - tol) && v < (expected + tol);
  };

  float f = sycl::sqrt(x);
  bool b = near(f,x_sqrt);

  const unsigned sz{8};
  bool *p = new bool[sz]{}; // zeroed (to false)
  sycl::range<1> r{sz};

  sycl::queue q;

  {  // buffer zone

    sycl::buffer<bool,1> buf{p, sz};

    q.submit([&](sycl::handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      sycl::accessor acc{ buf, cgh, sycl::write_only, sycl::no_init };
      cgh.parallel_for(r, [=](sycl::id<1> i) {
#else
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<Foo>(r, [=](sycl::id<1> i) {
#endif
        const auto near = [tol](const auto& v, const auto &expected) {
          return v > (expected - tol) && v < (expected + tol);
        };

        float f = sycl::sqrt(x);
        bool b = near(f,x_sqrt);
        acc[i] = b;
      });
    });

  }  // buffer destroyed

  b = b && p[0];

  delete [] p;

  return b;
}

int main(int argc, char *argv[])
{
  assert(test_maths(0.01f));
  return 0;
}
