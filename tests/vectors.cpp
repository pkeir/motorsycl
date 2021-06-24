#define SYCL_SIMPLE_SWIZZLE

// Section 4.14.2.3 Swizzles
// "Note that the simple swizzle functions are only available for up to 4
// element vectors and are only available when the macro SYCL_SIMPLE_SWIZZLES
// is defined before including <sycl/sycl.hpp>."

#if __has_include (<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cassert>
#include <type_traits>
#include <iostream>

bool no_implicit_conversion(const sycl::int4 &) { return true; }

int main(int argc, char *argv[])
{
  const float one{1}, two{2}, three{3}, four{4};
  sycl::float4 f4{one,two,three,four};
  // "For example auto x = f4.x() would not be valid." Section 4.14.2.4.
  float x1 = f4.x(); float x2 = f4.r(); float x3 = f4.s0();
  float y1 = f4.y(); float y2 = f4.g(); float y3 = f4.s1();
  float z1 = f4.z(); float z2 = f4.b(); float z3 = f4.s2();
  float w1 = f4.w(); float w2 = f4.a(); float w3 = f4.s3();
  assert(x1 == one   && x2 == one   && x3 == one);
  assert(y1 == two   && y2 == two   && y3 == two);
  assert(z1 == three && z2 == three && z3 == three);
  assert(w1 == four  && w2 == four  && w3 == four);
  auto res1 = f4 + f4;
  auto res2 = f4 - f4;
  static_assert(std::is_same_v<decltype(res1),sycl::float4>);
  static_assert(std::is_same_v<decltype(res2),sycl::float4>);
  float x4 = res1.x(); float x5 = res2.x();
  float y4 = res1.y(); float y5 = res2.y();
  float z4 = res1.z(); float z5 = res2.z();
  float w4 = res1.w(); float w5 = res2.w();
  assert(x4==one+one && y4==two+two && z4==three+three && w4==four+four);
  assert(x5==one-one && y5==two-two && z5==three-three && w5==four-four);

  sycl::float4 f4b{1,2,3,4}; // No C++11 narrowing errors
  sycl::float4 f4c(1);       // set all 4 elements to 1
  f4b += f4c;
  f4b += 1;
  f4b = f4b + 1;
#if defined(__MOTORSYCL__) || defined(TRISYCL_CL_LANGUAGE_VERSION)
  f4b = 1 + f4b;
#else
  f4b = 1.0f + f4b;
#endif
  f4b = 8.0f - f4b; // (8,8,8,8) - (5,6,7,8)
#ifdef TRISYCL_CL_LANGUAGE_VERSION
  no_implicit_conversion(43); // implicit conv prevented by ctor's explicit
#endif
  float x6 = f4b.x(); float y6 = f4b.y(); float z6 = f4b.z();float w6 = f4b.w();
  assert(x6==3 && y6==2 && z6==1 && w6==0);

  sycl::vec<int,1> i1{1};
  int x = i1;
  i1+1;
  static_assert(std::is_same_v<decltype(i1+1),sycl::vec<int,1>>);

#ifndef __MOTORSYCL__
{
  // 4.14.2.4. Swizzled vec class
  sycl::float4 sw = f4.swizzle<1,3,0,2>(); // auto not permitted
  float x1 = sw.x(); float x2 = sw.r(); float x3 = sw.s0();
  float y1 = sw.y(); float y2 = sw.g(); float y3 = sw.s1();
  float z1 = sw.z(); float z2 = sw.b(); float z3 = sw.s2();
  float w1 = sw.w(); float w2 = sw.a(); float w3 = sw.s3();
  assert(x1 == two   && x2 == two   && x3 == two);
  assert(y1 == four  && y2 == four  && y3 == four);
  assert(z1 == one   && z2 == one   && z3 == one);
  assert(w1 == three && w2 == three && w3 == three);

  sycl::vec<int,1> i1{42};
  int i = i1;
  assert(i == 42);
  static_assert(std::is_convertible_v<sycl::vec<int,1>,int>);
}
#endif
  return 0;
}
