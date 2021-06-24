#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

// * See https://github.com/intel/llvm/issues/3101

int main(int argc, char *argv[])
{
  using namespace sycl;
  const unsigned sz{1024};
  unsigned *p = new unsigned[sz]{}; // zeroed
  range<1> r{sz};

  queue q;

  {  // buffer zone

    buffer<unsigned,1> buf{p, sz};

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(r, [=](id<1> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(r, [=](id<1> i) {
#endif
        acc[i] = 9;
      });
    });

  }  // buffer destroyed

  q.wait(); // n.b. This does not guarantee that the data is copied back (*)

  assert(p[0]==9);
  delete [] p; // This *cannot* be freed b4 the buffer dtor (more obvious) (*)

  return 0;
}
