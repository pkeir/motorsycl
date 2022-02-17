#include <sycl/sycl.hpp>
#include <iostream>

template <typename T>
T* byte_offset(T* p, const unsigned offset = 1)
{
  char *pc       = reinterpret_cast<char*>(p);
  T    *p_offset = reinterpret_cast<T*>(pc+offset); // misaligned
  return p_offset;
}

template <unsigned sz>
bool use_host_ptr(const unsigned offset, const sycl::property_list& ps = {})
{
  using namespace sycl;

  int* a = new int[sz+offset]{};
  int* pao = byte_offset(a,offset);
  int* p{};

  {
    buffer<int, 1> buf{pao, sz, ps};

    {
      host_accessor h_acc(buf);
      p = h_acc.get_pointer();
    }
  }

  delete [] a;
  return pao == p;
}

int main(int argc, char *argv[])
{
  const unsigned sz{1024};

  const sycl::property_list ps = {sycl::property::buffer::use_host_ptr()};

  bool b1 = use_host_ptr<sz>(0);
  bool b2 = use_host_ptr<sz>(0, ps);
  bool b3 = use_host_ptr<sz>(1);
  bool b4 = use_host_ptr<sz>(1, ps);

#ifndef __COMPUTECPP__
  assert(b1);
#else
  assert(!b1);
#endif
  assert(b2);
  assert(!b3);
  assert(b4);

  return 0;
}
