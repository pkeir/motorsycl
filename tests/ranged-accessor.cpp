#include <iostream>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[])
{
  using namespace sycl;
  const unsigned sz{16};
  bool data[sz]{};
  bool* underlying[2]{};

  queue q;

  {
    buffer buf{&data[0], range<1>{sz}};
    buffer bufu{&underlying[0], range<1>{2}};

    // This assigns to 8 elements from the buffer, starting at
    // offset 4, using a ranged accessor
    q.submit([&](handler& cgh)
    {
      range<1> sub_r{8};
      id<1> offset{4};
      accessor acc{ buf, cgh, sub_r, offset };
      accessor accu{ bufu, cgh };

      cgh.parallel_for(range<1>{sub_r}, [=](item<1> i) {
        if (i[0]==0) {
          accu[0] = acc.get_pointer();
        }
        acc[i] = true;
      });
    });

    // This submission assigns to the 3rd element of the full buffer accessor
    // Intel's DPCPP indicates that space for the full buffer was already
    // allocated, as the addresses obtained simply precede those from above.
    // But Intel's get_pointer method is wrong here, as it should return
    // the "underlying buffer regardless of the accessor's offset" ... which
    // would be the same as that from above.
    q.submit([&](handler& cgh)
    {
      accessor acc{ buf, cgh };
      accessor accu{ bufu, cgh };

      cgh.parallel_for(range<1>{sz}, [=](item<1> i) {
        if (i[0]==2) {
          accu[1] = acc.get_pointer();
          acc[i] = true;
        }
      });
    });
  }

  for (int i = 0; i < sz; ++i)
     std::cout << data[i] << ' ';
  std::cout << '\n';

#ifdef __MOTORSYCL__
  // should return the "underlying buffer regardless of the accessor's offset"
  assert(underlying[0]==underlying[1]);
#else
  assert(underlying[0]!=underlying[1]);
#endif

  bool ideal[sz]{0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0};
  assert(std::equal(std::begin(data), std::end(data), std::begin(ideal)));

  return 0;
}
