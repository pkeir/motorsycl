#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

bool single_task_test()
{
  using namespace sycl;
  const unsigned sz{1};
  using T = int;
  T *data = new T[sz]{}; // zero the data
  queue q;

  {
    buffer<T,1> buf{ data, range<1>{sz} };

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.single_task([=]() {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<Foo>([=]() {
#endif
        acc[0] = 42;
      });
    });
  }

  bool b = data[0] == 42;

  delete [] data;
  return b;
}

int main(int argc, char *argv[])
{
  assert(single_task_test());
  return 0;
}
