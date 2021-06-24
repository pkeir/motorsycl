#ifndef _ZIP_WITH_HPP_
#define _ZIP_WITH_HPP_

#include <array>
#include <type_traits>
#include <utility>
#include <cstddef>

namespace sycl::detail
{

#if !defined(__clang__) && !defined(__NVCOMPILER)
template <
  typename R_ = void,
  typename F,
  typename T,
  std::size_t N,
  typename... As
>
constexpr auto zip_with(const F f, const std::array<T,N>& a, const As&... as)
{
  auto vred = [&]<std::size_t I>() { return f(a[I],as[I]...); };

  auto iseq = [&]<typename I, I... Is>(std::integer_sequence<I,Is...>) {
    using R = std::conditional_t<std::is_same_v<R_,void>,std::array<T,N>,R_>;
    return R{vred.template operator()<Is>()...};
  };

  return iseq(std::make_index_sequence<N>{});
}
#else

template <typename R_, typename I, I... Is, typename F, typename T, std::size_t N>
auto iseq(std::integer_sequence<I,Is...>,
          F f, const std::array<T,N>& a, const std::array<T,N>& b) {
  using R = std::conditional_t<std::is_same_v<R_,void>,std::array<T,N>,R_>;
  return R{f(a[Is],b[Is])...};
};

template <
  typename R_ = void,
  typename F,
  typename T,
  std::size_t N,
  typename... As
>
constexpr auto zip_with(const F f,
                        const std::array<T,N>& a, const std::array<T,N>&b)
{
  return iseq<R_>(std::make_index_sequence<N>{},f,a,b);
}
#endif

} // namespace sycl::detail

#endif // _ZIP_WITH_HPP_
