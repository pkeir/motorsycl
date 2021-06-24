#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

// $MYCUDA/bin/nvc++ -stdpar -std=c++17 -I include unused_buffer.cpp

bool destroy_test()
{
  using namespace sycl;

  const unsigned sz{1024};
  int *data1 = new int[sz]{}; // zero-initialised
  int *data2 = new int[sz]{}; // zero-initialised
  auto *pbuf1 = new buffer<int>{ data1, range<1>{sz} };
  auto *pbuf2 = new buffer<int>{ data2, range<1>{sz} };

  queue q;
  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor acc1{ *pbuf1, cgh, write_only, no_init };
//    cgh.parallel_for(sz, [=](id<1> i) {   // todo: look at range<1> ctors
    cgh.parallel_for(range{sz}, [=](id<1> i) {
#else
    auto acc1 = pbuf1->get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo>(sz, [=](id<1> i) {
#endif
      acc1[i] = i[0];
    });
  });

  delete pbuf2; // contains no reference to a sycl::queue
 
  bool ok2 = true;
  for (unsigned i = 0; i < sz; ++i) { ok2 = ok2 && data2[i]==0; } // unchanged
  
  delete pbuf1; // contains a valid sycl::queue: so calls q.wait();

  bool ok1 = true;
  for (unsigned i = 0; i < sz; ++i) { ok1 = ok1 && data1[i]==i; }

  delete [] data1; // correctl freed after the buffers
  delete [] data2;

  return ok1 && ok2;
}

int main(int argc, char *argv[])
{
  assert(destroy_test());
  return 0;
}
