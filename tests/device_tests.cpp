#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

struct custom_device_selector
{
  int operator()(const sycl::device& dev) const
  {
    using device_type = sycl::info::device::device_type;
    if (dev.get_info<device_type>() == sycl::info::device_type::cpu) {
      return 50;
    }
    if (dev.get_info<device_type>() == sycl::info::device_type::gpu) {
      return 100;
    }
    return -1; // Devices with a negative score will never be chosen.
  }
};

bool selector_test()
{
  using namespace sycl;
  const unsigned h{4}, w{2}, sz{h*w};
  int *data = new int[sz]{}; // zero the data

  property_list pl{property::buffer::use_host_ptr{}};
  custom_device_selector sel;
  queue q{sel}; // No: sel implicitly creates a property_list;
                // use C++20 concepts in the property_list constructors

  {
    buffer<int, 2> buf{ data, range<2>{h,w} };

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(range<2>{h,w}, [=](id<2> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(range<2>{h,w}, [=](id<2> i) {
#endif
        acc[i] = i[0]*w + i[1];
      });
    });
  }

  bool b = 0==data[0] && 1==data[1] && 2==data[2] && 3==data[3] &&
           4==data[4] && 5==data[5] && 6==data[6] && 7==data[7];
  delete [] data;
  return b;
}

int main(int argc, char *argv[])
{
  assert(selector_test());
  return 0;
}
