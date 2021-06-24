#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

bool multi_ptr_test()
{
  using namespace sycl;
  const unsigned sz{16};
  using T = int;
  T *data = new T[sz]{}; // zero the data
  queue q;

  {
    buffer<T, 1> buf{ data, range<1>{sz} };

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(range<1>{sz}, [=](id<1> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(range<1>{sz}, [=](id<1> i) {
#endif
#if defined(__MOTORSYCL__)
        auto ptr = sycl::make_ptr<T,access::address_space::global_space,
                                      access::decorated::legacy>(
                                        acc.get_pointer());
#endif
        using gptr_t = sycl::global_ptr<T>;
        gptr_t gptr(acc.get_pointer());
#ifndef TRISYCL_CL_LANGUAGE_VERSION
        T* p = gptr.get();
#else
        T *p = gptr;
#endif
        *(p + i[0]) = i[0];
#if SYCL_LANGUAGE_VERSION >= 202001
        //static_assert(std::is_same_v<typename gptr_t::value_type,T>);
        //value_type still absent from DPCPP
#endif

        T *raw_ptr = nullptr;
        gptr_t gptr2(raw_ptr);
        gptr2 = raw_ptr;

        //sycl::global_ptr<void> gptr3((void *)raw_ptr);
        //gptr3 = (void *)raw_ptr;
      });
    });
  }

  bool b{true};
  for (unsigned i = 0; i < sz; ++i) {
    b = b && data[i]==i;
  }

  delete [] data;
  return b;
}

int main(int argc, char *argv[])
{
  assert(multi_ptr_test());
  return 0;
}
