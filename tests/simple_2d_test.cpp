#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

// With clang this gives an error, probably a divide by zero on i/i

bool simple_2d_test()
{
  using namespace sycl;
  const unsigned h{4}, w{2}, sz{h*w};
  int *data = new int[sz]{}; // zero the data

  queue q;

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

#if !defined(__COMPUTECPP__) && !defined(TRISYCL_CL_LANGUAGE_VERSION)
        id<1> q;
        int x = 1 + q; // 1 + q returns an id<1>
#endif

        id<2> j{2,3};
        i += j; i -= j;
        i += 1; i *= 2; i /= 2; i -= 1;
// SYCL 2020:
#if defined(__MOTORSYCL__)
        i = -i; i = -i;
        ++i; --i;
        i++; i--;
#endif
        id<2> k1{j}, k2{id<2>{}};
        k1 = k2;
        k2 = id<2>{};
        k2 = k1 + id<2>{};
        j = j==j ? j : j;
        j = j!=j ? j : j;
        j = (j+j) + (j-j) + (j*j) + (j/j) + (j%j) + (j<<j) + (j>>j) + (j&j);
        j = (j|j) + (j^j) + (j&&j) + (j||j) + (j<j) + (j>j) + (j<=j) + (j>=j);
#ifndef __COMPUTECPP__
        j = (j+1) + (j-1) + (j*1) + (j/1) + (j%1) + (j<<1) + (j>>1) + (j&1);
#else
        j = (j+1) + (j-1) + (j*1) + (j/1) + (j%1) + (j<<1) + (j>>1);
#endif
        j = (j|1) + (j^1) + (j&&1) + (j||1) + (j<1) + (j>1) + (j<=1) + (j>=1);
        acc[i] = i[0]*w + i[1];
      });
    });
  }

  //q.wait();

  bool b = 0==data[0] && 1==data[1] && 2==data[2] && 3==data[3] &&
           4==data[4] && 5==data[5] && 6==data[6] && 7==data[7];
  delete [] data;
  return b;
}

int main(int argc, char *argv[])
{
  assert(simple_2d_test());
  return 0;
}
