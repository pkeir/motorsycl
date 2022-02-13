#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo1;
class Foo2;
class Foo2b;
class Foo3;

bool item_test()
{
  using namespace sycl;
  const unsigned sz{16};
  int *data = new int[sz]{}; // zero the data
  queue q;

  {
    buffer<int, 1> buf{ data, range<1>{sz} };

    q.submit([&](handler& cgh)
    {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(range<1>{sz}, [=](item<1> i) {
//      cgh.parallel_for(range<1>{sz}, [=](id<1> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo1>(range<1>{sz}, [=](item<1> i) {
#endif
        id<1> check{i}; // test id's item constructor
        item<1> i2{i}; // item copy ctor
        item<1> i3{std::move(i2)}; // item move ctor
        auto zero = i==i && !(i!=i) ? 0 : 1; // test item's operator==
        acc[i] = i.get_range(0) + i[0] + zero;
      });
    });
  }

  bool b{true};
  for (unsigned i = 0; i < sz; ++i) {
    b = b && data[i]==sz+i;
  }

  delete [] data;
  return b;
}

bool item_test_parallel_for_offsets()
{
  using namespace sycl;
  const unsigned sz{16}, sub_sz{8}, offset{3};
  int *data = new int[sz]{}; // zero the data
  queue q;

  {
    buffer<int, 1> buf{ data, range<1>{sz} };

    q.submit([&](handler& cgh)
    {
      range<1> sub_r{sub_sz};
      id<1> o{offset};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(sub_r, o, [=](item<1> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo2>(sub_r, o, [=](item<1> i) { // 3,4...
#endif
        id<1> check{i};
        acc[i] = i[0];
      });
    });
  }

  bool b{true};
  for (unsigned i = 0; i < sub_sz; ++i) { b = b && data[i+offset]==i+offset; }
  for (unsigned i = 0; i < offset; ++i) { b = b && data[i]==0; }
  for (unsigned i = offset+sub_sz; i < sz; ++i) { b = b && data[i]==0; }

  delete [] data;
  return b;
}

bool item_test_parallel_for_offsets_2d()
{
  using namespace sycl;
  const unsigned sz1{5}, sz2{6}, sub_sz1{3}, sub_sz2{4}, offset1{1}, offset2{1};
  int *data = new int[sz1*sz2]{}; // zero the data
  const int nine = 9;
  queue q;

  {
    buffer<int, 2> buf{ data, range<2>{sz1,sz2} };

    q.submit([&](handler& cgh)
    {
      range<2> sub_r{sub_sz1, sub_sz2};
      id<2> o{offset1, offset2};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(sub_r, o, [=](item<2> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo2b>(sub_r, o, [=](item<2> i) { // 3,4...
#endif
        acc[i.get_id()] = nine; // there is no accessor::operator[](item)?
      });
    });
  }

  int checksum{0};
  for (unsigned i = 0; i < sz1; ++i) {
    for (unsigned j = 0; j < sz2; ++j) {
      checksum += data[i*sz2+j];
    }
  }

  bool b{true};
  for (unsigned i = 0; i < sz1; ++i) {
  for (unsigned j = 0; j < sz2; ++j) {
    if (i>=offset1 && i<sub_sz1+offset1 && j>=offset2 && j<sub_sz2+offset2)
      b = b && data[i*sz2+j]==0;
    else
      b = b && data[i*sz2+j]==nine;
  }}

  delete [] data;
  return (checksum==(sub_sz1*sub_sz2*nine));
}

#ifndef TRISYCL_CL_LANGUAGE_VERSION
bool item_test_accessor_offsets()
{
  using namespace sycl;
  const unsigned sz{16}, sub_sz{8}, offset1{2}, offset2{3};
  int *data = new int[sz]{}; // zero the data
  queue q;

  {
    buffer<int, 1> buf{ data, range<1>{sz} };

    q.submit([&](handler& cgh)
    {
      range<1> sub_r{sub_sz};
      id<1> o1{offset1}, o2{offset2};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      //accessor acc{ buf, cgh, write_only, no_init };
      accessor acc{ buf, cgh, sub_r, o1 }; // use this one
      cgh.parallel_for(sub_r, o2, [=](item<1> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh, sub_r, o1); // 2,3...
      cgh.parallel_for<Foo3>(sub_r, o2, [=](item<1> i) { // 3,4...
#endif
        id<1> check{i};
        acc[i] = i[0];
      });
    });
  }

  bool b{true};
  for (unsigned i = 0; i < sub_sz; ++i) {
    b = b && data[i+offset1+offset2]==i+offset2;
  }
  for (auto i = 0; i < offset1+offset2; ++i) { b = b && data[i]==0; }
  for (auto i = offset1+offset2+sub_sz; i < sz; ++i) { b = b && data[i]==0; }

  delete [] data;
  return b;
}
#endif

int main(int argc, char *argv[])
{
  assert(item_test());
  assert(item_test_parallel_for_offsets());
  assert(item_test_parallel_for_offsets_2d());
#ifndef TRISYCL_CL_LANGUAGE_VERSION
  assert(item_test_accessor_offsets());
#endif
  return 0;
}
