#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>
#include <numeric>
#include <array>

class Foo;

int main(int argc, char *argv[])
{
  using namespace sycl;
  const unsigned sz{1024};
  std::array<int, sz> arr_in, arr_out;
  range<1> r{sz};

  std::iota(arr_in.begin(), arr_in.end(), 0);
  /*const*/ std::array<int, sz>& arr_in_ref = arr_in; // const ... buffer ctor.

  default_selector sel;
  queue q{sel};

  {  // buffer zone

    buffer buf_in{arr_in_ref};
    buffer buf_out{arr_out};

    q.submit([&](handler& cgh)
    {                      // __SYCL_COMPILER_VERSION covers the Intel compiler
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc_in{ buf_in, cgh, read_only };
      accessor acc_out{ buf_out, cgh, write_only, no_init };
      cgh.parallel_for(r, [=](id<1> i) {
#else
      auto acc_in = buf_in.get_access<access::mode::read>(cgh);
      auto acc_out = buf_out.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(r, [=](id<1> i) {
#endif
        acc_out[i] = acc_in[i];
      });
    });

  }  // buffer destroyed

  assert(arr_in==arr_out);

  return 0;
}
