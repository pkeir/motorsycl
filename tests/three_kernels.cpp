#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo1;
class Foo2;
class Foo3;

int main(int argc, char *argv[])
{
  using namespace sycl;

  const size_t N = 2000;
  buffer<int> a { range<1>{N} };
  buffer<int> b { range<1>{N} };
  buffer<int> c { range<1>{N} };

  queue q;
  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor A { a, cgh, write_only };
    cgh.parallel_for(range<1> {N}, [=](id<1> index) {
#else
    auto A = a.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo1>(range<1> {N}, [=](id<1> index) {
#endif
      A[index] = index[0];
    });
  });

  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor B { b, cgh, write_only };
    cgh.parallel_for(range<1> {N}, [=](id<1> index) {
#else
    auto B = b.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo2>(range<1> {N}, [=](id<1> index) {
#endif
      B[index] = index[0];
    });
  });

  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor A { a, cgh, read_only };
    accessor B { b, cgh, read_only };
    accessor C { c, cgh, write_only };
    cgh.parallel_for(range<1>{N}, [=](id<1> index) {
#else
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo3>(range<1>{N}, [=](id<1> index) {
#endif
      C[index] = A[index] + B[index];
    });
  });

#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
  host_accessor C { c, read_only };
#else
  auto C = c.get_access<access::mode::read>();
#endif

  bool ok = true;
  for (unsigned i = 0; i < N; ++i) {
    ok = ok && C[i]==i+i;
  }

  assert(ok);
  return 0;
}
