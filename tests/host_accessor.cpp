#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo1;
class Foo2;
class Foo3;

bool host_buffer_test1()
{
  using namespace sycl;
  const unsigned h{4}, w{3}, sz{w*h};
  int *data = new int[sz]{0,1,2,3,4,5,6,7,8,9,10,11};

  buffer<int, 1> buf_1d{ data, range<1>{sz} };
  buffer<int, 2> buf_2d{ data, range<2>{h,w} };
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
  host_accessor h_acc_1d { buf_1d, read_only };
  host_accessor h_acc_2d { buf_2d, read_only };
#else
  auto h_acc_1d = buf_1d.get_access<access::mode::read>();
  auto h_acc_2d = buf_2d.get_access<access::mode::read>();
#endif
  bool b1{true};
  for (unsigned i = 0; i < sz; ++i)
    b1 = b1 && h_acc_1d[i]==data[i];

  bool b2{true};
  unsigned count{0};
  for (unsigned i = 0; i < h; ++i) {
    for (unsigned j = 0; j < w; ++j) {
      b2 = b2 && data[count]==h_acc_2d[i][j];
      b2 = b2 && data[count]==h_acc_2d[id<2>{i,j}];
      ++count;
    }
  }

  delete [] data;
  return b1 && b2;
}

bool host_buffer_test2()
{
  using namespace sycl;
  const unsigned w{32}, h{16}, sz=w*h;
  int *data = new int[sz]{}; // zero the data

  queue q;
  buffer<int, 1> buf_1d{ data, range<1>{sz} };
  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor acc{ buf_1d, cgh, write_only, no_init };
    cgh.parallel_for(range<1>{sz}, [=](id<1> i) { acc[i] = i[0]; });
#else
    auto acc = buf_1d.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo1>(range<1>{sz}, [=](id<1> i) { acc[i] = i[0]; });
#endif
  });

  bool b{true};
  int count{0};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
  host_accessor h_acc_1d { buf_1d, read_only };
#else
  auto h_acc_1d = buf_1d.get_access<access::mode::read>();
#endif
  for (unsigned i = 0; i < sz; ++i) {
    b = b && h_acc_1d[i]==data[i] && data[count]==count;
    ++count;
  }

  delete [] data;
  return b;
}

bool host_buffer_test3()
{
  using namespace sycl;

  const unsigned w{32}, h{16}, sz{h*w};
  int *data = new int[sz]{}; // zero the data

  queue q;
  buffer<int, 2> buf_2d{ data, range<2>{h,w} };
  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor acc{ buf_2d, cgh, write_only, no_init };
    cgh.parallel_for(range<2>{h,w}, [=](id<2> i) {
      acc[i] = i[0]*w + i[1];
    });
#else
    auto acc = buf_2d.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo2>(range<2>{h,w}, [=](id<2> i){
      acc[i] = i[0]*w + i[1];
    });
#endif
  });

  bool b{true};
  int count{0};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
  host_accessor h_acc_2d { buf_2d, read_only };
#else
  auto h_acc_2d = buf_2d.get_access<access::mode::read>();
#endif
  for (unsigned i = 0; i < h; ++i) {
  for (unsigned j = 0; j < w; ++j) {
    b = b && h_acc_2d[i][j]==data[count] && data[count]==count;
    ++count;
  }}

  delete [] data;
  return b;
}

bool host_buffer_test4()
{
  using namespace sycl;

  const unsigned d{4}, h{8}, w{16}, sz{d*h*w};
  int *data = new int[sz]{}; // zero the data

  queue q;
  buffer<int, 3> buf_3d{ data, range<3>{d,h,w} };
  q.submit([&](handler& cgh)
  {
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
    accessor acc{ buf_3d, cgh, write_only, no_init };
    cgh.parallel_for(range<3>{d,h,w}, [=](id<3> i) {
      acc[i] = i[0]*h*w + i[1]*w + i[2];
    });
#else
    auto acc = buf_3d.get_access<access::mode::write>(cgh);
    cgh.parallel_for<Foo3>(range<3>{d,h,w}, [=](id<3> i){
      acc[i] = i[0]*h*w + i[1]*w + i[2];
    });
#endif
  });

  bool b{true};
  int count{0};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
  host_accessor h_acc_3d { buf_3d, read_only };
#else
  auto h_acc_3d = buf_3d.get_access<access::mode::read>();
#endif
  for (unsigned i = 0; i < d; ++i) {
  for (unsigned j = 0; j < h; ++j) {
  for (unsigned k = 0; k < w; ++k) {
    b = b && h_acc_3d[i][j][k]==data[count] && data[count]==count;
    ++count;
  }}}

  delete [] data;
  return b;
}

int main(int argc, char *argv[])
{
  assert(host_buffer_test1());
  assert(host_buffer_test2());
  assert(host_buffer_test3());
  assert(host_buffer_test4());
  return 0;
}
