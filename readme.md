# MotörSYCL

MotörSYCL (MotorSYCL) is a prototype Khronos SYCL implementation; a header-only library for the nvc++ compiler (from NVIDIA's HPC SDK).

The following program code is derived from the small test suite included in the
`tests` subdirectory. The program demonstrates use of Unified Shared Memory
(USM) via *device allocations*. Similar functionality for USM *shared
allocations* and *system allocations* can be seem in `tests/usm_shortcuts.cpp`.


```cpp
#include <sycl/sycl.hpp>
#include <cassert>
#include <numeric>

// USM Device Allocations

// nvc++ -stdpar -std=c++20 -I include example.cpp

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

  free(d_p, q); // sycl::free

  return 0;
}
```

The code above comes from the `example.cpp` file; found in the same
directory as this readme. The command required to build it is:

```
nvc++ -stdpar -std=c++20 -Wno-deprecated-declarations -I include example.cpp
```
