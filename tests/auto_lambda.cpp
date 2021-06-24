#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

// $MYCUDA/bin/nvc++ -stdpar -std=c++2a -I ../include auto_lambda.cpp

template <int dims, typename A>
void smokeout(sycl::id<dims> &i, A& acc) { if (0==i[0]) acc[i]=42; }

template <int I, bool B, typename A>
void smokeout(sycl::item<I,B> &i, A& acc) { if (0==i[0]) acc[i]=43; }

#ifdef __COMPUTECPP__
// ComputeCpp error without this, but the result is still not 44
template <typename A>
void smokeout(sycl::detail::item_base &i, A& acc)
{
  if (0==i[0]) acc[i]=44;
}
#endif

bool auto_lambda()
{
  using namespace sycl;
  unsigned sz{1024};
  range<1> r{sz};
  unsigned *p = new unsigned[sz]{}; // zeroed

  queue q;
  {
    buffer<unsigned,1> buf{p, sz};

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(r, [=](auto i) {
        unsigned value = 0;
        acc[i] = value + i;
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(r, [=](item<1> i) {
#endif
        smokeout(i,acc);
      });
    });
  }

  bool is_item = p[0]==43; // sycl::item on DPCPP, ComputeCpp
  delete [] p;

  return is_item;
}

int main(int argc, char *argv[])
{
  assert(auto_lambda());
  return 0;
}
