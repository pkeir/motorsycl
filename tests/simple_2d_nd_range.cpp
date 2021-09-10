#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>

class Foo;

bool simple_2d_nd_range()
{
  using namespace sycl;
  const unsigned gh{8}, gw{4}, sz{gh*gw}; // global range
  const unsigned lh{4}, lw{2};            // local (work-group) range
  bool *bdata = new bool[sz]{}; // zero the data (to false)

  queue q;

  {
    buffer<bool, 2> buf{ bdata, range<2>{gh,gw} };

    q.submit([&](handler& cgh)
    {
      range<2> gr{gh, gw};
      range<2> lr{lh, lw};
#if defined(__MOTORSYCL__) || defined(__SYCL_COMPILER_VERSION)
      accessor acc{ buf, cgh, write_only, no_init };
      cgh.parallel_for(nd_range<2>{gr,lr}, [=](nd_item<2> i) {
#else
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<Foo>(nd_range<2>{gr,lr}, [=](nd_item<2> i) {
#endif
        std::size_t linear_id = i.get_global_linear_id();
        id<2> global_id = i.get_global_id();
        std::size_t d0 = i.get_global_id(0);
        std::size_t d1 = i.get_global_id(1);
#if defined(TRISYCL_CL_LANGUAGE_VERSION)
        bool b1 = d1*gw + d0 == linear_id;
#else
        bool b1 = d0*gw + d1 == linear_id; // Correct (3.11.1 Linearization)
#endif
        acc[global_id] = b1;
      });
    });
  }

  //q.wait();

  bool b = true;
  for (std::size_t i = 0; i < sz; ++i)
    b = b && bdata[i];
  delete [] bdata;
  return b;
}

bool basic_nd_range_type()
{
  sycl::range<2> g1{13,5};   // n.b. here the local range doesn't evenly divide
  sycl::range<2> l1{4,2};    // the global range: get_group_range discards the
  sycl::id<2> o1{0,0};       // relevant remainders.
  sycl::range<2> gr1{g1[0]/l1[0],g1[1]/l1[1]};
  sycl::nd_range<2> nd{g1,l1};

  const auto g2 = nd.get_global_range();
  const auto l2 = nd.get_local_range();
  const auto o2 = nd.get_offset();
  const auto gr2 = nd.get_group_range();

  bool b = g1==g2 && l1==l2 && o1==o2 && gr1==gr2;
  bool b2 = nd == nd && !(nd != nd);

  return b && b2;
}

#if defined(__MOTORSYCL__)
bool basic_nd_item_type()
{
  sycl::range<2> g1{12,4};
  sycl::range<2> l1{4,2};
  sycl::nd_range<2> nd{g1,l1};

//  sycl::nd_item<2> i{sycl::id<2>{6,3},nd}; // no such constructor exists
  sycl::nd_item<2> i = sycl::detail::mk_nd_item(sycl::id<2>{6,3},nd);
  bool b = sycl::id<2>{6,3}==i.get_global_id();
  b = b && sycl::id<2>{2,1}==i.get_local_id();
  b = b && 2==i.get_local_id(0) && 1==i.get_local_id(1);
  b = b && sycl::id<2>{12,4}==i.get_global_range();
  b = b && 12==i.get_global_range(0) && 4==i.get_global_range(1);
  b = b && sycl::id<2>{4,2}==i.get_local_range();
  b = b && 4==i.get_local_range(0) && 2==i.get_local_range(1);
  return b;
}
#endif

int main(int argc, char *argv[])
{
  assert(basic_nd_range_type());
#if defined(__MOTORSYCL__)
  assert(basic_nd_item_type());
#endif
  assert(simple_2d_nd_range());
  return 0;
}
