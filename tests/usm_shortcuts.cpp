#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>
#include <numeric>
#include <iterator>

class Foo;

// $MYCUDA/bin/nvc++ -stdpar -acc -std=c++20 -I ../include usm_shortcuts.cpp
// 4.8. Unified shared memory (USM)

bool test_usm_device()
{
  using namespace sycl;
  const unsigned sz{1024};
  range<1> r{sz};
  queue q;

  bool b = q.get_device().has(aspect::usm_device_allocations);
  if (!b)
    return true;
  unsigned *d_p = malloc_device<unsigned>(sz,q); // USM pointer
  unsigned h[sz];
  std::iota(std::begin(h), std::end(h), 0);

  q.memcpy(d_p, h, sz * sizeof(unsigned));
  q.parallel_for(r, [=](id<1> i) { d_p[i] += i; });
  q.wait();
  q.memcpy(h, d_p, sz * sizeof(unsigned));

  for (unsigned i = 0; i < sz; ++i)
    b = b && (h[i] == 2*i);

  free(d_p,q); // sycl::free
  return b;
}

bool test_usm_shared()
{
  using namespace sycl;
  unsigned sz{1024};
  range<1> r{sz};
  queue q;

  bool b = q.get_device().has(aspect::usm_shared_allocations);
  if (!b)
    return true;
  unsigned *p = malloc_shared<unsigned>(sz,q); // USM pointer

  q.parallel_for(r, [=](id<1> i) { p[i] = i; });
  q.wait();

  for (unsigned i = 0; i < sz; ++i)
    b = b && (p[i] == i);

  free(p,q);
  return b;
}

bool test_usm_system()
{
  using namespace sycl;
  unsigned sz{1024};
  range<1> r{sz};
  queue q;

  bool b = q.get_device().has(aspect::usm_system_allocations);
  if (!b)
    return true;
  unsigned *p = new unsigned[sz];

  q.parallel_for(r, [=](id<1> i) { p[i] = i; });
  q.wait();

  for (unsigned i = 0; i < sz; ++i)
    b = b && (p[i] == i);

  delete [] p;
  return b;
}

int main(int argc, char *argv[])
{
  assert(test_usm_device());
  assert(test_usm_shared());
  assert(test_usm_system());
  return 0;
}
