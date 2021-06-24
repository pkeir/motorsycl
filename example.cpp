#include <sycl/sycl.hpp>
#include <cassert>
#include <numeric>

// USM Device Allocations

// $MYNVCPP/bin/nvc++ -stdpar -std=c++20 -I include example.cpp

int main(int argc, char *argv[])
{
  using namespace sycl;
  const unsigned sz{1024};
  range<1> r{sz};
  queue q;

  bool b = q.get_device().has(aspect::usm_device_allocations);
  assert(b);

  unsigned *d_p = malloc_device<unsigned>(sz,q); // USM pointer
  unsigned h[sz];
  std::iota(std::begin(h), std::end(h), 0);

  q.memcpy(d_p, h, sz * sizeof(unsigned));
  q.parallel_for(r, [=](id<1> i) { d_p[i] += i; });
  q.wait();
  q.memcpy(h, d_p, sz * sizeof(unsigned));

  for (unsigned i = 0; i < sz; ++i)
    b = b && (h[i] == 2*i);

  assert(b);

  free(d_p,q); // sycl::free
  return 0;
}

