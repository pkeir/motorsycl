#ifndef __ITERQ_HPP__
#define __ITERQ_HPP__

#include <sycl/sycl.hpp>
#include <iterator> // std::ptrdiff_t
#include <array>    // std::array
#include <tuple>    // std::tuple

namespace sycl::detail {

template <typename> class iter;
template <typename, bool> class iterq;

// These type parameters really need to be std::array, so make that explicit?
template <typename Ti, typename Tr>
class iter<std::tuple<Ti,Tr>>
{
  using tuple2_t = std::tuple<Ti,Tr>;

protected:

  using value_type = Ti;

  iter() : i_{get<0>(x_)}, ub_{get<1>(x_)} {}
  iter(const iter& o) : x_{o.x_}, i_{get<0>(x_)}, ub_{get<1>(x_)} {}
  iter(const tuple2_t& i) : x_{i}, i_{get<0>(x_)}, ub_{get<1>(x_)} {}

  iter& operator=(const iter& o) { x_ = o.x_; return *this; }

  template <typename U>
  auto dist(U i) const { return i_[i]; }

  template <typename U>
  auto range(U i) const { return ub_[i]; }

  value_type& dump_i() { return get<0>(dump_); }

  tuple2_t x_;
  Ti& i_;
  Tr& ub_;
  tuple2_t dump_;
};

template <typename Ti, typename Tr, typename To>
class iter<std::tuple<Ti,Tr,To>>
{
  using tuple3_t = std::tuple<Ti,Tr,To>;

protected:

  using value_type = Ti;

  iter() : i_{get<0>(x_)}, lb_{get<2>(x_)} {}
  iter(const iter& o) : x_{o.x_}, i_{get<0>(x_)}, lb_{get<2>(x_)}, ub_{o.ub_} {}
  iter(const tuple3_t& i)
    : x_{i}, i_{get<0>(x_)}, lb_{get<2>(x_)},
      ub_{sycl::detail::zip_with<Tr>(std::plus{},get<2>(i),get<1>(i))} {}

  iter& operator=(const iter& o)
  { x_ = o.x_; ub_ = o.ub_; return *this; }

  template <typename U>
  auto dist(U i) const { return i_[i] - lb_[i]; }

  template <typename U>
  auto range(U i) const { return ub_[i] - lb_[i]; }

  value_type& dump_i() { return get<0>(dump_); }

  tuple3_t x_;
  Ti& i_;
  To& lb_;
  Tr ub_;
  tuple3_t dump_;
};

template <int dims>
class iter<sycl::item<dims,false>>
{
  using item_t = sycl::item<dims,false>;

protected:

  using value_type = sycl::id<dims>;

  iter() : i_{x_.id_}, ub_{x_.range_} {}
  iter(const iter& o) : x_{o.x_}, i_{x_.id_}, ub_{x_.range_} {}
  iter(const item_t& i) : x_{i}, i_{x_.id_}, ub_{x_.range_} {}

  iter& operator=(const iter& o) { x_ = o.x_; return *this; }

  template <typename U>
  auto dist(U i) const { return i_[i]; }

  template <typename U>
  auto range(U i) const { return ub_[i]; }

  value_type& dump_i() { return dump_.id_; }

  item_t x_;
  sycl::id<dims>& i_;
  sycl::range<dims>& ub_;
  item_t dump_;
};

template <int dims>
class iter<sycl::item<dims,true>>
{
  using item_t = sycl::item<dims,true>;

protected:

  using value_type = sycl::id<dims>;

  iter() : i_{x_.id_}, lb_{x_.offset_} {}
  iter(const iter& o) : x_{o.x_}, i_{x_.id_}, lb_{x_.offset_}, ub_{o.ub_} {}
  iter(const item_t& i) : x_{i}, i_{x_.id_}, lb_{x_.offset_},
    ub_{sycl::detail::zip_with<sycl::range<dims>>(std::plus{}, i.offset_,i.range_)} {}

  iter& operator=(const iter& o)
  { x_ = o.x_; ub_ = o.ub_; return *this; }

  template <typename U>
  auto dist(U i) const { return i_[i] - lb_[i]; }

  template <typename U>
  auto range(U i) const { return ub_[i] - lb_[i]; }

  value_type& dump_i() { return dump_.id_; }

  item_t x_;
  sycl::id<dims>& i_;
  sycl::id<dims>& lb_;
  sycl::range<dims> ub_;
  item_t dump_;
};

template <int dims>
class iter<sycl::nd_item<dims>>
{
  using nd_item_t = sycl::nd_item<dims>;

protected:

  using value_type = sycl::id<dims>;

  iter() : i_{x_.id_}, lb_{x_.nd_range_.offset_} {}
  iter(const iter& o)
    : x_{o.x_}, i_{x_.id_}, lb_{x_.nd_range_.offset_}, ub_{o.ub_} {}
  iter(const nd_item_t& i) : x_{i}, i_{x_.id_}, lb_{x_.nd_range_.offset_},
    ub_{sycl::detail::zip_with<sycl::range<dims>>(std::plus{},i.nd_range_.offset_,i.nd_range_.global_range_)} {}

  iter& operator=(const iter& o)
  { x_ = o.x_; ub_ = o.ub_; return *this; }

  template <typename U>
  auto dist(U i) const { return i_[i] - lb_[i]; }

  template <typename U>
  auto range(U i) const { return ub_[i] - lb_[i]; }

  value_type& dump_i() { return dump_.id_; }

  nd_item_t x_;
  sycl::id<dims>& i_;
  sycl::id<dims>& lb_;
  sycl::range<dims> ub_;
  nd_item_t dump_;
};

template <typename T, bool use_aggregate = true>
struct iterq : public iter<T>
{
  using difference_type = std::ptrdiff_t;
  using value_type =
    std::conditional_t<use_aggregate,T,typename iter<T>::value_type>;
  using reference = const value_type&;
  using pointer = const value_type*;
  using iterator_category = std::random_access_iterator_tag;

private:

  using self = iterq;

  difference_type flatten() const
  {
    difference_type flat_index{0}, accum{1};
    for (auto i{std::size(this->i_)}; i-- > 0;)
    {
      flat_index += this->dist(i) * accum;
      accum *= this->range(i);
    }

    return flat_index;
  }

  template <typename U, size_t dims>
  void add(std::array<U,dims>& res, difference_type n)
  {
    difference_type carry{0};
    for (auto i{dims-1}; i > 0; --i)
    {
     const auto rem = n % this->range(i);
     const auto x = res[i] + rem + carry; // candidate
     res[i] = (x < this->ub_[i]) ? (carry=0,x) : (carry=1,x-this->range(i));
     n = n / this->range(i);
    }
    res[0] = res[0] + n + carry;
  }

public:

  iterq() : iter<T>{} {}

  template <int dims, bool with_offset>
  iterq(const sycl::item<dims,with_offset>& i) : iter<T>{i} { }

  template <int dims>
  iterq(const sycl::nd_item<dims>& i) : iter<T>{i} { }

  template <typename Ui, typename Ur>
  iterq(const Ui& i, const Ur& r) : iter<T>{T{i,r}} {}

  template <typename Ui, typename Ur, typename Uo>
  iterq(const Ui& i, const Ur& r, const Uo& o) : iter<T>{T{i,r,o}} {}

  reference operator*() const noexcept
  {
    if constexpr (use_aggregate) return this->x_;
    else return this->i_;
  }
  self& operator++() noexcept
  {
    add(this->i_, 1);
    return *this;
  } // pre-incr
  self operator++(int) noexcept { self t{*this}; ++(*this); return t; }

  auto operator<=>(const self& rhs) const {
    using array_t = const class iter<T>::value_type::array;
    return static_cast<array_t>(this->i_) <=> static_cast<array_t>(rhs.i_);
  }

  bool operator==(const self& rhs) const {
    using array_t = const class iter<T>::value_type::array;
    return static_cast<array_t&>(this->i_) == static_cast<array_t&>(rhs.i_);
  }

  // bidirectional iterator:
  self& operator--() noexcept
  {
    add(this->i_, -1);
    return *this;
  } // pre-decr
  self operator--(int) noexcept { self t{*this}; --(*this); return t; }

  // random access iterator:
  self &operator+=(const difference_type n)
  {
    add(this->i_, n);
    return *this;
  }
  self operator+(const difference_type n) const
  {
    self t{*this};
    return t += n;
  }
  self &operator-=(const difference_type n)
  {
    add(this->i_, -n);
    return *this;
  }
  self operator-(const difference_type n) const
  {
    self t{*this};
    return t -= n;
  }
  reference operator[](const difference_type n)
  {
    this->dump_ = this->x_;
    add(this->dump_i(), n);
    if constexpr (use_aggregate) return this->dump_;
    else return this->dump_i();
  }
  difference_type operator-(const self &other) const {
    return this->flatten() - other.flatten();
  }

  pointer operator->() const noexcept
  {
    if constexpr (use_aggregate) return &this->x_;
    else return &this->i_;
  }
};

template <typename T>
iterq(const T&) -> iterq<T,true>;

template <typename Ui, typename Ur>
iterq(const Ui& i, const Ur& r) -> iterq<std::tuple<Ui,Ur>,false>;

template <typename Ui, typename Ur, typename Uo>
iterq(const Ui& i, const Ur& , const Uo&) -> iterq<std::tuple<Ui,Ur,Uo>,false>;

} // namespace sycl::detail

#endif // __ITERQ_HPP__
