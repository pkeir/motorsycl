#ifndef _MOTORSYCL_HPP_
#define _MOTORSYCL_HPP_

// Copyright (c) 2020-2021 Paul Keir, University of the West of Scotland.

#define __MOTORSYCL__
#define SYCL_LANGUAGE_VERSION 202001

#include <cuda_runtime.h>
#include <cstddef>       // std::size_t
#include <memory>        // std::allocator
#include <type_traits>   // std::remove_const_t
#include <queue>         // std::queue
#include <functional>    // std::function, std::plus
#include <cassert>       // assert
#include <algorithm>     // std::for_each
#include <numeric>       // std::iota (temporarily)
#include <optional>      // std::optional
#include <tuple>         // std::tuple_size, std::tuple_element
#include <array>         // std::array
#include <execution>     // std::execution Requires -ltbb
#include <cmath>         // std::sin, std::cos, std::sqrt
#include <mutex>         // std::mutex
#include <span>          // std::span, std::dynamic_extent
#include <bit>           // std::bit_cast
#include <exception>     // std::exception_ptr
#include <system_error>  // std::is_error_condition_enum
#include <unordered_map> // std::unordered_map
#include <variant>       // std::variant

namespace sycl
{

class device;
class context;
class exception_list;
template <typename T>
using buffer_allocator = std::allocator<T>;
template <
  typename T,
  int dims = 1,
  typename AllocT = buffer_allocator<std::remove_const_t<T>>
> requires(dims==1||dims==2||dims==3) class buffer;
template <int dims = 1> requires(dims==1||dims==2||dims==3) class range;
template <int dims = 1> requires(dims==1||dims==2||dims==3) class nd_range;
template <int dims = 1> requires(dims==1||dims==2||dims==3) class id;
template <int dims = 1> requires(dims==1||dims==2||dims==3) class group;
template <int = 1, bool with_offset = true> class item;
template <int = 1> class nd_item;
template <typename, int> class vec;
enum class aspect;

// Section 4.7.6.1 Access targets
enum class target {
  device,
  host_task,
  global_buffer = device, // Deprecated
  constant_buffer,        // Deprecated
  local,                  // Deprecated
  host_buffer             // Deprecated
};
namespace access {
  using sycl::target;
} // namespace access

// Section 4.7.6.2
enum class access_mode {
  read,
  write,
  read_write,
  discard_write,      // Deprecated in SYCL 2020
  discard_read_write, // Deprecated in SYCL 2020
  atomic              // Deprecated in SYCL 2020
};
namespace access {
  using sycl::access_mode;
}

// Section 4.7.6.5 Placeholder accessor
enum class placeholder { // Deprecated in SYCL 2020
  false_t,
  true_t,
};
namespace access {
  using sycl::placeholder;
}

// Section 4.7.6.9.1 Interface for buffer command accessors
template <
  typename dataT,
  int dims = 1,
  access_mode AccessMode = (std::is_const_v<dataT> ? access_mode::read
                                                   : access_mode::read_write),
  target AccessTarget = target::device,
  access::placeholder isPlaceholder = access::placeholder::false_t // *
  // * Deprecated in SYCL 2020:
>
class accessor;

// Section 4.7.6.10.1 Interface for buffer host accessors
template <
  typename dataT,
  int dims = 1,
  access_mode AccessMode = (std::is_const_v<dataT> ? access_mode::read :
                                                     access_mode::read_write)
>
class host_accessor;

namespace property {

class no_init;

namespace buffer {

class use_host_ptr;
class use_mutex;
class context_bound;

} // namespace property
} // namespace buffer

// Section 4.5.4.1 Properties interface
template <typename Property>
struct is_property : std::false_type {};

template <typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;

template <typename Property, typename SyclObject>
struct is_property_of;

template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v =
  is_property_of<Property, SyclObject>::value;

template <>
struct is_property<property::buffer::use_host_ptr>  : std::true_type {};

namespace detail {

template <typename> struct is_buffer   : std::false_type {};
template <typename> struct is_accessor : std::false_type {};
template <typename T, int dims, typename AllocT>
struct is_buffer<buffer<T,dims,AllocT>> : std::true_type {};
template <
  typename dataT,
  int dims,
  access_mode AccessMode,
  target AccessTarget,
  access::placeholder isPlaceholder
>
struct is_accessor<accessor<dataT,dims,AccessMode,AccessTarget,isPlaceholder>>
  : std::true_type {};

} // namespace detail

template <typename T>
struct is_property_of<property::buffer::use_host_ptr,T>
  : std::disjunction<detail::is_buffer<T>> {}; // disjunction? for other classes

template <>
struct is_property<property::buffer::use_mutex>     : std::true_type {};

template <typename T>
struct is_property_of<property::buffer::use_mutex,T>
  : std::disjunction<detail::is_buffer<T>> {};

template <>
struct is_property<property::buffer::context_bound> : std::true_type {};

template <typename T>
struct is_property_of<property::buffer::context_bound,T>
  : std::disjunction<detail::is_buffer<T>> {};

template <>
struct is_property<property::no_init>               : std::true_type {};

template <typename T>
struct is_property_of<property::no_init,T>
  : std::disjunction<detail::is_accessor<T>> {};

} // namespace sycl

#include "sycl/detail/pgi.hpp"
#include "sycl/detail/zip_with.hpp"
namespace sycl {

// Section 3.9.2 Alignment with future versions of C++
using std::span;
using std::dynamic_extent;
#ifndef __NVCOMPILER
using std::bit_cast;
#endif

namespace detail {
  const auto g_pol = std::execution::par_unseq;
  //const auto g_pol = std::execution::seq;
}

enum class backend : char { host, nvhpc };

// Section 4.5.1.1. Type traits backend_traits
template <backend>
  class backend_traits;

template <>
class backend_traits<backend::host>
{
  struct todo_backend_host {};

public:

  template <class T>
  using input_type = todo_backend_host;

  template <class T>
  using return_type = todo_backend_host;

  using errc = todo_backend_host;
};

template <>
class backend_traits<backend::nvhpc>
{
  struct todo_backend_nvhpc {};

public:

  template <class T>
  using input_type = todo_backend_nvhpc;

  template <class T>
  using return_type = todo_backend_nvhpc;

  using errc = todo_backend_nvhpc;
};

template <backend Backend, typename SyclType>
using backend_input_t =
  typename backend_traits<Backend>::template input_type<SyclType>;

template <backend Backend, typename SyclType>
using backend_return_t =
  typename backend_traits<Backend>::template return_type<SyclType>;

// Appendix A: Information descriptors
namespace info {

// Section A.1. Platform information descriptors
namespace platform {

struct profile;
struct version;
struct name;
struct vendor;
struct extensions; // Deprecated

} // namespace platform

// A.2. Context information descriptors
namespace context {

struct platform;
struct devices;
struct atomic_memory_order_capabilities;
struct atomic_fence_order_capabilities;
struct atomic_memory_scope_capabilities;
struct atomic_fence_scope_capabilities;

} // namespace context

enum class device_type : unsigned int
{
  cpu, // Maps to OpenCL CL_DEVICE_TYPE_CPU
  gpu, // Maps to OpenCL CL_DEVICE_TYPE_GPU
  accelerator, // Maps to OpenCL CL_DEVICE_TYPE_ACCELERATOR
  custom, // Maps to OpenCL CL_DEVICE_TYPE_CUSTOM
  automatic, // Maps to OpenCL CL_DEVICE_TYPE_DEFAULT
  host,
  all // Maps to OpenCL CL_DEVICE_TYPE_ALL
};

// Section A.3. Device information descriptors
// Section 4.6.4.2. Device information descriptors
namespace device {

struct device_type { using return_type = info::device_type; };
struct vendor_id { using return_type = uint32_t; };
struct max_compute_units { using return_type = uint32_t; };
struct max_work_item_dimensions { using return_type = uint32_t; };
template<int dimensions = 3> struct max_work_item_sizes;
template<> struct max_work_item_sizes<1> { using return_type = id<1>; };
template<> struct max_work_item_sizes<2> { using return_type = id<2>; };
template<> struct max_work_item_sizes<3> { using return_type = id<3>; };
struct max_work_group_size { using return_type = std::size_t; };
struct preferred_vector_width_char;
struct preferred_vector_width_short;
struct preferred_vector_width_int;
struct preferred_vector_width_long;
struct preferred_vector_width_float;
struct preferred_vector_width_double;
struct preferred_vector_width_half;
struct native_vector_width_char;
struct native_vector_width_short;
struct native_vector_width_int;
struct native_vector_width_long;
struct native_vector_width_float;
struct native_vector_width_double;
struct native_vector_width_half;
struct max_clock_frequency;
struct address_bits;
struct max_mem_alloc_size;
struct image_support; // Deprecated
struct max_read_image_args;
struct max_write_image_args;
struct image2d_max_height;
struct image2d_max_width;
struct image3d_max_height;
struct image3d_max_width;
struct image3d_max_depth;
struct image_max_buffer_size;
struct max_samplers;
struct max_parameter_size;
struct mem_base_addr_align;
struct half_fp_config;
struct single_fp_config;
struct double_fp_config;
struct global_mem_cache_type;
struct global_mem_cache_line_size;
struct global_mem_cache_size;
struct global_mem_size;
struct max_constant_buffer_size; // Deprecated
struct max_constant_args; // Deprecated
struct local_mem_type;
struct local_mem_size;
struct error_correction_support;
struct host_unified_memory;
struct atomic_memory_order_capabilities;
struct atomic_fence_order_capabilities;
struct atomic_memory_scope_capabilities;
struct atomic_fence_scope_capabilities;
struct profiling_timer_resolution;
struct is_endian_little;
struct is_available;
struct is_compiler_available; // Deprecated
struct is_linker_available; // Deprecated
struct execution_capabilities;
struct queue_profiling; // Deprecated
struct built_in_kernels; // Deprecated
struct built_in_kernel_ids;
struct platform { using return_type = platform; };
struct name { using return_type = std::string; };
struct vendor { using return_type = platform; };
struct driver_version { using return_type = platform; };
struct profile { using return_type = platform; };
struct version { using return_type = platform; };
struct backend_version { using return_type = platform; };
struct aspects;
struct extensions; // Deprecated
struct printf_buffer_size;
struct preferred_interop_user_sync;
struct parent_device;
struct partition_max_sub_devices;
struct partition_properties;
struct partition_affinity_domains;
struct partition_type_property;
struct partition_type_affinity_domain;

} // namespace device

enum class partition_property : int
{
  no_partition,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

enum class partition_affinity_domain : int
{
  not_applicable,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : int {
  denorm,
  inf_nan,
  round_to_nearest,
  round_to_zero,
  round_to_inf,
  fma,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, read_write };

enum class execution_capability : unsigned int
{
  exec_kernel,
  exec_native_kernel
};

// Section A.4. Queue information descriptors
// Section 4.6.5.3. Queue information descriptors
namespace queue {

struct context { using return_type = ::sycl::context; };
struct device { using return_type = ::sycl::device; };

} // namespace queue

// Section A.5. Kernel information descriptors
namespace kernel {

struct num_args;
struct attributes;

} // namespace kernel

namespace kernel_device_specific {

struct global_work_size;
struct work_group_size;
struct compile_work_group_size;
struct preferred_work_group_size_multiple;
struct private_mem_size;
struct max_num_sub_groups;
struct compile_num_sub_groups;
struct max_sub_group_size;
struct compile_sub_group_size;

} // namespace kernel_device_specific

// Section A.6. Event information descriptors
namespace event {

struct command_execution_status;

} // namespace event

enum class event_command_status : int {
  submitted,
  running,
  complete
};

namespace event_profiling {

struct command_submit;
struct command_start;
struct command_end;

} // namespace event_profiling

} // namespace info

/*inline std::ostream& operator<<(std::ostream &o, backend be)
{
  switch (be)
  {
    case backend::host:
      o << "host";
      break;
    case backend::nvhpc:
      o << "nvhpc";
  }

  return o;
}*/

// Section 4.6.2.1 Platform interface
// Glossary: "A collection of devices managed by a single backend."
class platform
{
public:

  platform() : backend_{backend::nvhpc} { }

  template <typename DeviceSelector>
  explicit platform(const DeviceSelector &sel) { assert(0); }

  /* -- common reference semantics -- */

  platform(const platform&)                                = default;
  platform(const platform&&)                               = default;
  platform& operator=(platform&&)                          = default;
  platform& operator=(const platform&)                     = default;
  ~platform()                                              = default;
  friend bool operator==(const platform&, const platform&) = default;
  friend bool operator!=(const platform&, const platform&) = default;

  backend get_backend() const { return backend_; }

  std::vector<device>
  get_devices(info::device_type dt = info::device_type::all) const
  {
    assert(dt==info::device_type::all);
    return devices_;
  }

  template <typename Param>
  typename Param::return_type get_info() const
  { assert(0); return {}; }

  template <typename Param>
  typename Param::return_type get_backend_info() const
  { assert(0); return {}; }

  bool has(aspect asp) const
  { assert(0); return {}; }

  bool has_extension(const std::string &extension) const // Deprecated
  { assert(0); return {}; }

  static std::vector<platform> get_platforms()
  { assert(0); return {}; }

private:
  backend backend_;
  std::vector<device> devices_;
};

namespace detail {

template <typename...>
struct type_set : std::true_type {};

template <typename T, typename... Ts>
struct type_set<T,Ts...>
{
  static const bool value = (!std::is_same_v<T,Ts> && ...)
                            && type_set<Ts...>::value;
};

template <typename... Ts>
constexpr bool type_set_v = type_set<Ts...>::value;

} // namespace detail

class property_list
{
public:
  template <typename... Properties>
  requires std::conjunction_v<is_property<Properties>...> &&
           detail::type_set_v<Properties...>
  property_list(Properties... props) : ps_{{v_t{props}...}} {}

private:
  using v_t = std::variant<property::buffer::use_host_ptr,
                           property::buffer::use_mutex,
                           property::buffer::context_bound,
                           property::no_init>;

  std::vector<v_t> ps_;
};

using async_handler = std::function<void(sycl::exception_list)>;

namespace detail {
struct device_allocation {
  void* d_p_; // device data
  bool operator==(const device_allocation&) const = default; // for context ==
};
}

class context
{
  platform platform_;
  std::unordered_map<void*,detail::device_allocation> allocations_;

  template <typename, int, access_mode, target, access::placeholder>
  friend class accessor;

public:

  explicit context(const property_list &ps = {}) { }
  explicit context(async_handler ah, const property_list &ps = {})
  { assert(0); }
  explicit context(const device &dev, const property_list &ps = {})
  { assert(0); }
  explicit context(const device &dev, async_handler ah,
                   const property_list &ps = {}) { assert(0); }
  explicit context(const std::vector<device> &deviceList,
                   const property_list &ps = {}) { assert(0); }
  explicit context(const std::vector<device> &deviceList, async_handler ah,
                   const property_list &ps = {}) { assert(0); }

  /* -- property interface members -- */

  /* -- common reference semantics -- */

  context(const context &rhs) { assert(0); }
  context(context &&rhs) { assert(0); }
  context &operator=(const context &rhs) { assert(0); return *this; }
  context &operator=(context &&rhs) { assert(0); return *this; }
  ~context() {
    for (const auto& a : allocations_)
      cudaFree(a.second.d_p_);
  }

  friend bool operator==(const context&, const context&) = default;
  friend bool operator!=(const context&, const context&) = default;

  backend get_backend() const noexcept { return platform_.get_backend(); }
  platform get_platform() const { return platform_; }
  std::vector<device> get_devices() const { return platform_.get_devices(); }

  template <typename param>
  typename param::return_type get_info() const
  { assert(0); }

  template <typename param>
  typename param::return_type get_backend_info() const
  { assert(0); }
};

namespace detail {
  context g_context{}; // default context
}

// Section 4.13.2. Exception class interface
class exception : public virtual std::exception
{
public:

  exception(std::error_code ec, const std::string& what_arg) { assert(0); }
  exception(std::error_code ec, const char * what_arg) { assert(0); }
  exception(std::error_code ec) { assert(0); }
  exception(int ev, const std::error_category& ecat,
            const std::string& what_arg) { assert(0); }
  exception(int ev, const std::error_category& ecat, const char* what_arg) {
    assert(0);
  }
  exception(int ev, const std::error_category& ecat) { assert(0); }
  exception(context ctx, std::error_code ec, const std::string& what_arg) {
    assert(0);
  }
  exception(context ctx, std::error_code ec, const char* what_arg) {
    assert(0);
  }
  exception(context ctx, std::error_code ec) { assert(0); }
  exception(context ctx, int ev, const std::error_category& ecat,
            const std::string& what_arg) { assert(0); }
  exception(context ctx, int ev, const std::error_category& ecat,
            const char* what_arg) { assert(0); }
  exception(context ctx, int ev, const std::error_category& ecat) {
    assert(0);
  }
  const std::error_code& code() const noexcept {
    assert(0);
    static const std::error_code err;
    return err;
  }
  const std::error_category& category() const noexcept {
    assert(0);
    return std::system_category();
  }
  // PGK: Added "noexcept":  https://stackoverflow.com/a/53830534/2023370
  const char *what() const noexcept { assert(0); return "abcdefghijklm"; }
  bool has_context() const noexcept { assert(0); return {}; }
  context get_context() const { assert(0); context c; return c; }
};

// Used as a container for a list of asynchronous exceptions
class exception_list
{
  using container_t = std::vector<std::exception_ptr>;

public:

  using value_type = std::exception_ptr;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;
  using iterator = typename container_t::const_iterator;
  using const_iterator = typename container_t::const_iterator;

  size_type size() const { return data_.size(); }

  // first asynchronous exception
  iterator begin() { return data_.begin(); }

  // refer to past-the-end last asynchronous exception
  iterator end() { return data_.end(); }

private:
  container_t data_;
};

enum class errc
{
  runtime,
  kernel,
  accessor,
  nd_range,
  event,
  kernel_argument,
  build,
  invalid,
  memory_allocation,
  platform,
  profiling,
  feature_not_supported,
  kernel_not_supported,
  backend_mismatch
};

template <backend b>
using errc_for = typename backend_traits<b>::errc;

std::error_condition make_error_condition(errc e) noexcept {
  assert(0);
  return std::error_condition{};
}
std::error_code make_error_code(errc e) noexcept {
  assert(0);
  return std::error_code{};
}
const std::error_category& sycl_category() noexcept {
  assert(0);
  return std::system_category();
}

template<backend b>
const std::error_category& error_category_for() noexcept {
  assert(0);
  return std::system_category();
}

} // namespace sycl

namespace std {

template <>
struct is_error_condition_enum<sycl::errc> : std::true_type {};

template <>
struct is_error_code_enum<sycl::errc> : std::true_type {};

template <>
struct is_error_code_enum<sycl::backend_traits<sycl::backend::host>::errc>
  : std::true_type {};

template <>
struct is_error_code_enum<sycl::backend_traits<sycl::backend::nvhpc>::errc>
  : std::true_type {};

} // namespace std

namespace sycl {

// Section 4.6.6 Event class
class event
{
  cudaEvent_t ce_;
public:

  event() { cudaEventCreateWithFlags(&ce_, cudaEventDisableTiming); }

  /* -- common reference semantics -- */

  event(const event&)                                     = default;
  event(const event&&)                                    = default;
  event& operator=(event&&)                               = default;
  event& operator=(const event&)                          = default;
  ~event()                                                = default;
  friend bool operator==(const event &, const event &rhs) = default;
  friend bool operator!=(const event &, const event &rhs) = default;

  void wait() { cudaEventSynchronize(ce_); }
};

// Section 4.6.4.3. Device aspects
enum class aspect {
  cpu,
  gpu,
  accelerator,
  custom,
  emulated,
  host_debuggable,
  fp16,
  fp64,
  atomic64,
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_atomic_host_allocations,
  usm_shared_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations
};

// Section 4.6.4.1. Device interface
class device
{
public:

  device() : name_{"nvhpc"} { }

  template <typename DeviceSelector>
  explicit device(const DeviceSelector &sel)
  {
    const std::vector<device>& ds = get_devices();
    auto comp = [&](const auto& x, const auto& y) { return sel(x) < sel(y); };
    const auto it = std::max_element(ds.begin(), ds.end(), comp);
    if (it != ds.end()) { platform_ = it->get_platform(); }
  }

  /* -- common reference semantics -- */

  device(const device&)                                = default;
  device(const device&&)                               = default;
  device& operator=(device&&)                          = default;
  device& operator=(const device&)                     = default;
  ~device()                                            = default;
  friend bool operator==(const device&, const device&) = default;
  friend bool operator!=(const device&, const device&) = default;

  backend get_backend() const noexcept { return platform_.get_backend(); }

  bool is_cpu() const;

  bool is_gpu() const;

  bool is_accelerator() const;

  platform get_platform() const { return platform_; }

  template <typename param>
  typename param::return_type get_info() const
  {
    if constexpr (std::is_same_v<param,sycl::info::device::name>) {
      return name_;
    }
    else if constexpr (std::is_same_v<param,sycl::info::device::device_type>) {
      using P = decltype(detail::g_pol);
      if        (std::is_same_v<const P,decltype(std::execution::seq)>) {
        return info::device_type::cpu;
      } else if (std::is_same_v<const P,decltype(std::execution::par)>) {
        return info::device_type::cpu;
      } else if (std::is_same_v<const P,decltype(std::execution::par_unseq)>) {
        return info::device_type::cpu;
      }
    // The (C++17 only) nvc++ compiler has the C++20 unseq
    #if defined(__NVCOMPILER) || __cplusplus > 201703L
      else if (std::is_same_v<const P,decltype(std::execution::unseq)>) {
        return info::device_type::gpu;
      }
    #endif
    }
  }

  template <typename param>
  typename param::return_type get_backend_info() const;

  bool has(aspect asp) const {
         if (aspect::usm_device_allocations == asp) return true;
    else if (aspect::usm_shared_allocations == asp) return true;
    else if (aspect::usm_system_allocations == asp) return true;
    assert(0);
    return false;
  }

  [[deprecated]] bool has_extension(const std::string &extension) const;

  template <info::partition_property Prop>
  std::vector<device> create_sub_devices(size_t count)
    requires(Prop==info::partition_property::partition_equally)
  { assert(0); return {}; }

  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(const std::vector<size_t> &counts) const
    requires(Prop==info::partition_property::partition_by_counts)
  { assert(0); return {}; }

  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const
    requires(Prop==info::partition_property::partition_by_affinity_domain)
  { assert(0); return {}; }

  static std::vector<device>
  get_devices(info::device_type dt = info::device_type::all)
  {
    device host, nvhpc;
    host.name_ = "host";
    nvhpc.name_ = "nvhpc";
    std::vector<device> all_devices{host,nvhpc};
    return all_devices;
  }

private:
  std::string name_;
  platform platform_; // a reference to a platform, shared by multiple devices?
};

struct default_selector
{
  int operator()(const device& dev) const {
    return backend::nvhpc==dev.get_backend() ? 1 : 0;
  }
};

// The use of operator templates avoids say 1 + id becoming id + id via id's
// conversion operator, and so clashing with size_t + id (with Clang at least)
// Now, the operator template takes priority: 1 + id is matched as size_t + id

#define mk_bin_op(CLASS,OP) \
friend CLASS operator OP(const CLASS& a, const CLASS& b) { \
  return detail::zip_with<CLASS>([](const auto &x, const auto &y) \
    { return x OP y; }, a, b); \
} \
template <typename U> requires(std::integral<U>) \
friend CLASS operator OP(const CLASS& a, const U& b) { \
  return [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    return CLASS{(a[Is] OP b) ...}; \
  }(std::make_index_sequence<dims>{}); \
} \
template <typename U> requires(std::integral<U>) \
friend CLASS operator OP(const U& a, const CLASS& b) { \
  return [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    return CLASS{(a OP b[Is]) ...}; \
  }(std::make_index_sequence<dims>{}); \
}

#define mk_inc_by_op(CLASS,OP) \
friend CLASS& operator OP(CLASS& a, const CLASS& b) { \
  [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    ((a[Is] OP b[Is]), ...); \
  }(std::make_index_sequence<dims>{}); \
  return a; \
} \
friend CLASS& operator OP(CLASS& a, const size_t& b) { \
  [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    ((a[Is] OP b), ...); \
  }(std::make_index_sequence<dims>{}); \
  return a; \
}

#define mk_unary_op(CLASS,OP) \
friend CLASS operator OP(const CLASS& a) { \
  return [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    return CLASS{(OP a[Is]) ...}; \
  }(std::make_index_sequence<dims>{}); \
}

#define mk_inc_op(CLASS,OP) \
friend CLASS& operator OP(CLASS& a) { \
  [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    ((OP a[Is]), ...); \
  }(std::make_index_sequence<dims>{}); \
  return a; \
} \
friend CLASS& operator OP(CLASS& a, int) { \
  [&]<typename T, T... Is>(std::integer_sequence<T,Is...>) { \
    ((a[Is] OP), ...); \
  }(std::make_index_sequence<dims>{}); \
  return a; \
}

#define mk_bin_ops(CLASS) \
mk_bin_op(CLASS,+) \
mk_bin_op(CLASS,-) \
mk_bin_op(CLASS,*) \
mk_bin_op(CLASS,/) \
mk_bin_op(CLASS,%) \
mk_bin_op(CLASS,<<) \
mk_bin_op(CLASS,>>) \
mk_bin_op(CLASS,&) \
mk_bin_op(CLASS,|) \
mk_bin_op(CLASS,^) \
mk_bin_op(CLASS,&&) \
mk_bin_op(CLASS,||) \
mk_bin_op(CLASS,<) \
mk_bin_op(CLASS,>) \
mk_bin_op(CLASS,<=) \
mk_bin_op(CLASS,>=)

#define mk_inc_by_ops(CLASS) \
mk_inc_by_op(CLASS,+=) \
mk_inc_by_op(CLASS,-=) \
mk_inc_by_op(CLASS,*=) \
mk_inc_by_op(CLASS,/=) \
mk_inc_by_op(CLASS,%=) \
mk_inc_by_op(CLASS,<<=) \
mk_inc_by_op(CLASS,>>=) \
mk_inc_by_op(CLASS,&=) \
mk_inc_by_op(CLASS,|=) \
mk_inc_by_op(CLASS,^=)

#define mk_unary_ops(CLASS) mk_unary_op(CLASS,+) mk_unary_op(CLASS,-)
#define mk_inc_ops(CLASS) mk_inc_op(CLASS,++) mk_inc_op(CLASS,--)

// Section 4.9.1.1 range class
template <int dims>
requires(dims==1||dims==2||dims==3)
class range : public std::array<std::size_t, dims>
{
public:
  range() = default;
/*  range(std::size_t d, std::convertible_to<std::size_t> auto... ds)
    requires((sizeof...(ds))+1==dims)
    : std::array<std::size_t,dims>{d,static_cast<std::size_t>(ds)...} {}
// GCC bug 100138. Workaround below:
*/
  template <typename ...Ts>
  range(std::size_t d, Ts... ds)
    requires((sizeof...(Ts))+1==dims &&
              (std::convertible_to<Ts,std::size_t> && ...))
    : std::array<std::size_t,dims>{d,static_cast<std::size_t>(ds)...} {}

  size_t get(int dim) const { return this->operator[](dim); }
  size_t size() const {
    return std::apply([](const auto& ...xs) { return (xs * ...); },
                         static_cast<typename range::array>(*this));
  }

  mk_bin_ops(range);
  mk_inc_by_ops(range);
  mk_unary_ops(range);
  mk_inc_ops(range);
};

range(std::size_t) -> range<1>;
range(std::size_t, std::size_t) -> range<2>;
range(std::size_t, std::size_t, std::size_t) -> range<3>;

// Section 4.9.1.7. group class
template <int dims>
requires(dims==1||dims==2||dims==3)
class group : public std::array<std::size_t, dims>
{
public:
  using id_type = id<dims>;
  using range_type = range<dims>;
  using linear_id_type = std::size_t;

  static constexpr int dimensions = dims;
  //static constexpr memory_scope fence_scope = memory_scope::work_group;

  id<dims> get_group_id() const
  { assert(0); return {}; }
  size_t get_group_id(int dimension) const
  { assert(0); return {}; }
  id<dims> get_local_id() const
  { assert(0); return {}; }
  size_t get_local_id(int dimension) const
  { assert(0); return {}; }
  range<dims> get_local_range() const
  { assert(0); return {}; }
  size_t get_local_range(int dimension) const
  { assert(0); return {}; }
  range<dims> get_group_range() const
  { assert(0); return {}; }
  size_t get_group_range(int dimension) const
  { assert(0); return {}; }
  range<dims> get_max_local_range() const
  { assert(0); return {}; }
//  size_t operator[](int dimension) const
//  { assert(0); return {}; }
  size_t get_group_linear_id() const
  { assert(0); return {}; }
  size_t get_local_linear_id() const
  { assert(0); return {}; }
  size_t get_group_linear_range() const
  { assert(0); return {}; }
  size_t get_local_linear_range() const
  { assert(0); return {}; }
  bool leader() const
  { assert(0); return {}; }
  template<typename workItemFunctionT>
  void parallel_for_work_item(const workItemFunctionT &func) const
  { assert(0); }
  template<typename workItemFunctionT>
  void parallel_for_work_item(range<dimensions> logicalRange,
  const workItemFunctionT &func) const
  { assert(0); }

  /*
  template <typename dataT>
  device_event async_work_group_copy(decorated_local_ptr<dataT> dest,
  decorated_global_ptr<dataT> src, size_t numElements) const
  { assert(0); return {}; }
  template <typename dataT>
  device_event async_work_group_copy(decorated_global_ptr<dataT> dest,
  decorated_local_ptr<dataT> src, size_t numElements) const
  { assert(0); return {}; }

  template <typename dataT>
  device_event async_work_group_copy(decorated_local_ptr<dataT> dest,
  decorated_global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  { assert(0); return {}; }

  template <typename dataT>
  device_event async_work_group_copy(decorated_global_ptr<dataT> dest,
  decorated_local_ptr<dataT> src, size_t numElements, size_t destStride) const
  { assert(0); return {}; }
  */

  template <typename... eventTN>
  void wait_for(eventTN... events) const
  { assert(0); }
};

namespace detail {

template <int dims>
item<dims,false>
mk_item(const id<dims>& i, const range<dims>& r) {
  return {i,r};
}

template <int dims>
item<dims,true>
mk_item(const id<dims>& i, const range<dims>& r, const id<dims>& o) {
  return {i,r,o};
}

} // namespace detail

// Section 4.9.1.4 item class
template <int dims, bool>
class item
{
  id<dims> id_;
  range<dims> range_;
  id<dims> offset_;

  template <int, bool> friend class item;

  template <size_t, int dims_, bool with_offset_>
  friend auto& get(item<dims_, with_offset_>&);
  template <size_t, int dims_, bool with_offset_>
  friend const auto& get(const item<dims_, with_offset_>&);
  item() = default; // debug? remove ... yes, should be = delete
  //size_t& operator[](int dim) { return id_[dim]; }

  template <int D>
  friend item<D,true> detail::mk_item(const id<D>&, const range<D>&, const id<D>&); // debug

  friend class handler;
  item(const id<dims>& i, const range<dims>& range, const id<dims>& offset)
    : id_{i}, range_{range}, offset_{offset} {}

public:

  id<dims> get_id() const { return id_; }
  size_t get_id(int dim) const { return id_[dim]; }
  size_t operator[](int dim) const { return id_[dim]; }
  range<dims> get_range() const { return range_; }
  size_t get_range(int dim) const { return range_[dim]; }
  id<dims> get_offset() const { return offset_; }

  operator size_t() const requires(dims==1) { return id_[0]; }

  size_t get_linear_id() const
  {
    const auto& r = range_;
         if constexpr (dims==1) return id_[0];
    else if constexpr (dims==2) return id_[1] + id_[0]*r[1];
    else if constexpr (dims==3) return id_[2] + id_[1]*r[2] + id_[0]*r[1]*r[2];
  }

  // Common by-value semantics Section 4.5.3 Table 10
  friend bool operator==(const item& x, const item& y) {
    return x.id_==y.id_ && x.range_==y.range_ && x.offset_==y.offset_;
  }
  friend bool operator!=(const item& x, const item& y) { return !(x==y); }
};

template <int dims>
class item<dims,false>
{
  id<dims> id_;
  range<dims> range_;

  template <size_t, int dims_, bool with_offset_>
  friend auto& get(item<dims_, with_offset_>&);
  template <size_t, int dims_, bool with_offset_>
  friend const auto& get(const item<dims_, with_offset_>&);
  item() = default;
  //size_t& operator[](int dim) { return id_[dim]; }

  template <int D>
  friend item<D,false> detail::mk_item(const id<D>&, const range<D>&); // dbg

  friend class handler;
  item(const id<dims>& i, const range<dims>& range) : id_{i}, range_{range} {}

public:

  id<dims> get_id() const { return id_; }
  size_t get_id(int dim) const { return id_[dim]; }
  size_t operator[](int dim) const { return id_[dim]; }
  range<dims> get_range() const { return range_; }
  size_t get_range(int dim) const { return range_[dim]; }
  operator item<dims,true>() const { return {id_,range_,{}}; }

  operator size_t() const requires(dims==1) {
    static_assert(dims==1);
    return id_[0];
  }

  size_t get_linear_id() const {
    const auto& r = range_;
         if constexpr (dims==1) return id_[0];
    else if constexpr (dims==2) return id_[1] + id_[0]*r[1];
    else if constexpr (dims==3) return id_[2] + id_[1]*r[2] + id_[0]*r[1]*r[2];
  }

  friend bool operator==(const item& x, const item& y) {
    return x.id_==y.id_ && x.range_==y.range_;
  }
  friend bool operator!=(const item& x, const item& y) { return !(x==y); }
};

template <int dims>
requires(dims==1||dims==2||dims==3)
class id : public std::array<std::size_t, dims>
{
public:
  // Construct a SYCL id with the value 0 for each dimension.
  id() : std::array<std::size_t, dims>{} {}

  // common interface members (by-value semantics) (rule of zero)

/*  id(std::size_t d, std::convertible_to<std::size_t> auto... ds)
    requires((sizeof...(ds))+1==dims)
    : std::array<std::size_t,dims>{d,static_cast<std::size_t>(ds)...} {}
// GCC bug 100138. Workaround below:
*/

  template <typename ...Ts>
  id(std::size_t d, Ts... ds)
    requires((sizeof...(Ts))+1==dims &&
              (std::convertible_to<Ts,std::size_t> && ...))
    : std::array<std::size_t,dims>{d,static_cast<std::size_t>(ds)...} {}
  id(const range<dims> &range) : std::array<std::size_t,dims>{range} {}

  size_t get(int dim) const { return this->operator[](dim); }
  operator size_t() const requires(dims==1) { return (*this)[0]; }

  mk_bin_ops(id);
  mk_inc_by_ops(id);
  mk_unary_ops(id);
  mk_inc_ops(id);
};

// Deduction guides
id(std::size_t) -> id<1>;
id(std::size_t, std::size_t) -> id<2>;
id(std::size_t, std::size_t, std::size_t) -> id<3>;

// Section 4.9.1.2. nd_range class
template <int dims>
requires(dims==1||dims==2||dims==3)
class nd_range
{
  range<dims> global_range_;
  range<dims> local_range_;
  id<dims> offset_;

public:

  nd_range() = default;

  nd_range(range<dims> g, range<dims> l, id<dims> offset = id<dims>{})
    : global_range_{g}, local_range_{l}, offset_{offset} { }

  friend bool operator==(const nd_range& lhs, const nd_range& rhs) {
    return lhs.get_global_range() == rhs.get_global_range() &&
           lhs.get_local_range() == rhs.get_local_range(); }
  friend bool operator!=(const nd_range& lhs, const nd_range& rhs) {
    return !(lhs==rhs);
  }

  range<dims> get_global_range() const { return global_range_; }
  range<dims> get_local_range() const { return local_range_; }
  range<dims> get_group_range() const {
    return
    detail::zip_with<range<dims>>(std::divides{}, global_range_, local_range_);
  }
  id<dims> get_offset() const { return offset_; } // Deprecated SYCL 2020
};

} // namespace sycl

namespace sycl {

namespace detail {

template <int dims>
nd_item<dims>
mk_nd_item(const id<dims>& offset) { return {offset}; }

} // namespace detail

// Section 4.9.1.5 nd_item class
template <int dims>
class nd_item
{
  id<dims> offset_;

  nd_item(const id<dims>& offset) : offset_{offset} {}
  template <int d>
  friend nd_item<d> detail::mk_nd_item(const id<d>&);

public:

  id<dims>                get_global_id() const requires(dims==1)
  {
    return {_BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X};
  }
  id<dims>                get_global_id() const requires(dims==2)
  {
    return {_BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X,
            _BLOCKIDX_Y * _BLOCKDIM_Y + _THREADIDX_Y};
  }
  id<dims>                get_global_id() const requires(dims==3)
  {
    return {_BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X,
            _BLOCKIDX_Y * _BLOCKDIM_Y + _THREADIDX_Y,
            _BLOCKIDX_Z * _BLOCKDIM_Z + _THREADIDX_Z};
  }
  size_t                  get_global_id(int  ) const requires(dims==1)
  {
    return _BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X;
  }
  size_t                  get_global_id(int i) const requires(dims==2)
  {
    return i ? _BLOCKIDX_Y * _BLOCKDIM_Y + _THREADIDX_Y
             : _BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X;
  }
  size_t                  get_global_id(int i) const requires(dims==3)
  {
     return i ? (i==2 ? _BLOCKIDX_Z * _BLOCKDIM_Z + _THREADIDX_Z
                      : _BLOCKIDX_Y * _BLOCKDIM_Y + _THREADIDX_Y)
                      : _BLOCKIDX_X * _BLOCKDIM_X + _THREADIDX_X;
  }
  // Section 3.11.1 Linearization
  // ...also C.7.7. OpenCL kernel conventions and SYCL
  size_t                  get_global_linear_id() const requires(dims==1) {
    return get_global_id(0);
  }
  size_t                  get_global_linear_id() const requires(dims==2) {
    const auto id = get_global_id();
    const auto  r = get_global_range();
    return id[1] + (id[0] * r[1]);
  }
  size_t                  get_global_linear_id() const requires(dims==3) {
    const auto id = get_global_id();
    const auto  r = get_global_range();
    return id[2] + (id[1] * r[2]) + (id[0] * r[1] * r[2]);
  }
  id<dims>                get_local_id() const requires(dims==1)
  {
    return {_THREADIDX_X};
  }
  id<dims>                get_local_id() const requires(dims==2)
  {
    return {_THREADIDX_X, _THREADIDX_Y};
  }
  id<dims>                get_local_id() const requires(dims==3)
  {
    return {_THREADIDX_X, _THREADIDX_Y, _THREADIDX_Z};
  }
  size_t                  get_local_id(int ) const requires(dims==1)
  {
    return _THREADIDX_X;
  }
  size_t                  get_local_id(int i) const requires(dims==2)
  {
    return i ? _THREADIDX_Y : _THREADIDX_X;
  }
  size_t                  get_local_id(int i) const requires(dims==3)
  {
     return i ? (i==2 ? _THREADIDX_Z : _THREADIDX_Y) : _THREADIDX_X;
  }
  size_t                  get_local_linear_id() const
  { assert(0); return {}; }

  group<dims>             get_group() const requires(dims==1)
  { return {_BLOCKIDX_X}; }

  group<dims>             get_group() const requires(dims==2)
  { return {_BLOCKIDX_X, _BLOCKIDX_Y}; }

  group<dims>             get_group() const requires(dims==3)
  { return {_BLOCKIDX_X, _BLOCKIDX_Y, _BLOCKIDX_Z}; }

  size_t                  get_group(int  ) const requires(dims==1)
  { return _BLOCKIDX_X; }

  size_t                  get_group(int i) const requires(dims==2)
  { return i ? _BLOCKIDX_Y : _BLOCKIDX_X; }

  size_t                  get_group(int i) const requires(dims==3)
  { return i ? (i==2 ? _BLOCKIDX_Z : _BLOCKIDX_Y) : _BLOCKIDX_X; }

  size_t                  get_group_linear_id() const requires(dims==1)
  {
    return get_group(0);
  }

  size_t                  get_group_linear_id() const requires(dims==2)
  {
    const auto  g = get_group();
    const auto  r = get_group_range();
    return g[1] + (g[0] * r[1]);
  }

  size_t                  get_group_linear_id() const requires(dims==3)
  {
    const auto  g = get_group();
    const auto  r = get_group_range();
    return g[2] + (g[1] * r[2]) + (g[0] * r[1] * r[2]);
  }

  range<dims>             get_group_range() const requires(dims==1)
  { return {_GRIDDIM_X}; }

  range<dims>             get_group_range() const requires(dims==2)
  { return {_GRIDDIM_X, _GRIDDIM_Y}; }

  range<dims>             get_group_range() const requires(dims==3)
  { return {_GRIDDIM_X, _GRIDDIM_Y, _GRIDDIM_Z}; }

  size_t                  get_group_range(int  ) const requires(dims==1)
  { return _GRIDDIM_X; }

  size_t                  get_group_range(int i) const requires(dims==2)
  { return i ? _GRIDDIM_Y : _GRIDDIM_X; }

  size_t                  get_group_range(int i) const requires(dims==3)
  { return i ? (i==2 ? _GRIDDIM_Z : _GRIDDIM_Y) : _GRIDDIM_X; }

  range<dims>             get_global_range() const requires(dims==1)
  {
    return {_GRIDDIM_X * _BLOCKDIM_X};
  }
  range<dims>             get_global_range() const requires(dims==2)
  {
    return {_GRIDDIM_X * _BLOCKDIM_X,
            _GRIDDIM_Y * _BLOCKDIM_Y};
  }
  range<dims>             get_global_range() const requires(dims==3)
  {
    return {_GRIDDIM_X * _BLOCKDIM_X,
            _GRIDDIM_Y * _BLOCKDIM_Y,
            _GRIDDIM_Z * _BLOCKDIM_Z};
  }

  size_t                  get_global_range(int  ) const requires(dims==1)
  {
    return _GRIDDIM_X * _BLOCKDIM_X;
  }
  size_t                  get_global_range(int i) const requires(dims==2)
  {
    return i ? _GRIDDIM_Y * _BLOCKDIM_Y
             : _GRIDDIM_X * _BLOCKDIM_X;
  }
  size_t                  get_global_range(int i) const requires(dims==3)
  {
    return i ? (i==2 ? _GRIDDIM_Z * _BLOCKDIM_Z
                     : _GRIDDIM_Y * _BLOCKDIM_Y)
                     : _GRIDDIM_X * _BLOCKDIM_X;
  }

  range<dims>             get_local_range() const requires (dims==1)
  { return {_BLOCKDIM_X}; }

  range<dims>             get_local_range() const requires (dims==2)
  { return {_BLOCKDIM_X, _BLOCKDIM_Y}; }

  range<dims>             get_local_range() const requires (dims==3)
  { return {_BLOCKDIM_X, _BLOCKDIM_Y, _BLOCKDIM_Z}; }

  size_t                  get_local_range(int  ) const requires (dims==1)
  { return _BLOCKDIM_X; }

  size_t                  get_local_range(int i) const requires (dims==2)
  { return i ? _BLOCKDIM_Y : _BLOCKDIM_X; }

  size_t                  get_local_range(int i) const requires (dims==3)
  { return i ? (i==2 ? _BLOCKDIM_Z : _BLOCKDIM_Y) : _BLOCKDIM_X; }

  [[deprecated]] id<dims> get_offset() const
  { return offset_; }

  nd_range<dims>          get_nd_range() const
  { return {get_global_range(), get_local_range()}; }
};

// Section 4.7.6.3 Access tags
template <access_mode>
struct mode_tag_t {
  explicit mode_tag_t() = default;
};

inline constexpr mode_tag_t<access_mode::read> read_only{};
inline constexpr mode_tag_t<access_mode::read_write> read_write{};
inline constexpr mode_tag_t<access_mode::write> write_only{};

template <access_mode, target>
struct mode_target_tag_t {
  explicit mode_target_tag_t() = default;
};

inline constexpr
mode_target_tag_t<access_mode::read, target::constant_buffer> read_constant{};

// Section 4.7.6.9.2 Device buffer accessor properties
namespace property {
  struct no_init {};
} // namespace property

inline constexpr property::no_init no_init;

class queue;

namespace detail {

template <typename F, typename = void>
struct is_generic : std::true_type {};

template <typename F>
struct is_generic<F,std::void_t<decltype(&F::operator())>> : std::false_type {};

template <typename T>
constexpr auto is_generic_v = is_generic<T>::value;

} // namespace detail

// Section 4.7.7. Address space classes
// Section 4.7.7.1. Multi-pointer class
namespace access {

enum class address_space : int {
  global_space,
  local_space,
  constant_space, // Deprecated in SYCL 2020
  private_space,
  generic_space,
};

enum class decorated : int {
  no,
  yes,
  legacy,
};

} // namespace access

template <typename T> struct remove_decoration {
  using type = T;
};

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

template <
  typename ElementType,
  access::address_space Space,
  access::decorated DecorateAddress
>
class multi_ptr
{
public:
  static constexpr bool is_decorated = DecorateAddress==access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;

  using __unspecified__ = value_type; // todo
  using pointer = std::conditional_t<is_decorated, __unspecified__ *,
                                     std::add_pointer_t<value_type>>;
  using reference = std::conditional_t<is_decorated, __unspecified__ &,
                                     std::add_lvalue_reference_t<value_type>>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                std::add_pointer_t<value_type>>);
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                std::add_lvalue_reference_t<value_type>>);

  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr() {}
  multi_ptr(const multi_ptr& p) : m_p{p.m_p} {}
  multi_ptr(multi_ptr&& p) : multi_ptr{p} {}
  explicit multi_ptr(
    typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer)
  { assert(0); }
  multi_ptr(std::nullptr_t) : m_p{} {}

  // Only if Space == global_space or generic_space
  template <int dims, access_mode Mode, access::placeholder isPlaceholder>
  multi_ptr(accessor<value_type, dims, Mode, target::device, isPlaceholder>)
    requires(Space==access::address_space::local_space ||
             Space==access::address_space::generic_space)
  { assert(0); }

  // Only if Space == local_space or generic_space
  //template <int dims>
  //multi_ptr(local_accessor<ElementType, dims>);

  // Deprecated
  // Only if Space == local_space or generic_space
  template <int dims, access_mode Mode, access::placeholder isPlaceholder>
  multi_ptr(accessor<value_type, dims, Mode, target::local, isPlaceholder>);

  // Assignment and access operators

  multi_ptr &operator=(const multi_ptr&);
  multi_ptr &operator=(multi_ptr&&);
  multi_ptr &operator=(std::nullptr_t);

  // Only if Space == address_space::generic_space
  // and ASP != access::address_space::constant_space
  template<access::address_space ASP, access::decorated IsDecorated>
  multi_ptr &operator=(const multi_ptr<value_type, ASP, IsDecorated>&);
  // Only if Space == address_space::generic_space
  // and ASP != access::address_space::constant_space
  template<access::address_space ASP, access::decorated IsDecorated>
  multi_ptr &operator=(multi_ptr<value_type, ASP, IsDecorated>&&);

  reference operator*() const;
  pointer operator->() const;

  pointer get() const;
  std::add_pointer_t<value_type> get_raw() const;
  __unspecified__ * get_decorated() const;

  // Conversion to the underlying pointer type
  // Deprecated, get() should be used instead.
  operator pointer() const;

  // Only if Space == address_space::generic_space
  // Cast to private_ptr
  explicit operator multi_ptr<value_type, access::address_space::private_space,
                              DecorateAddress>();

  // Only if Space == address_space::generic_space
  // Cast to private_ptr
  explicit
  operator multi_ptr<const value_type, access::address_space::private_space,
                     DecorateAddress>() const;

  // Only if Space == address_space::generic_space
  // Cast to global_ptr
  explicit operator multi_ptr<value_type, access::address_space::global_space,
                              DecorateAddress>()
    requires(Space==access::address_space::generic_space);

  // Only if Space == address_space::generic_space
  // Cast to global_ptr
  explicit
  operator multi_ptr<const value_type, access::address_space::global_space,
                     DecorateAddress>() const;

  // Only if Space == address_space::generic_space
  // Cast to local_ptr
  explicit operator multi_ptr<value_type, access::address_space::local_space,
  DecorateAddress>();

  // Only if Space == address_space::generic_space
  // Cast to local_ptr
  explicit
  operator multi_ptr<const value_type, access::address_space::local_space,
                     DecorateAddress>() const;

  // Implicit conversion to a multi_ptr<void>.
  template <access::decorated DecorateAddress2>
  operator multi_ptr<void, Space, DecorateAddress2>() const
    requires(!std::is_const_v<value_type>)
  { assert(0); return {}; }

  // Implicit conversion to a multi_ptr<const void>.
  template <access::decorated DecorateAddress2>
  operator multi_ptr<const void, Space, DecorateAddress2>() const
    requires(std::is_const_v<value_type>)
  { assert(0); return {}; }

  // Implicit conversion to multi_ptr<const value_type, Space>.
  template <access::decorated DecorateAddress2>
  operator multi_ptr<const value_type, Space, DecorateAddress2>() const;

  // Implicit conversion to the non-decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::no>() const
    requires(is_decorated)
  { assert(0); return {}; }

  // Implicit conversion to the decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::yes>() const
    requires(!is_decorated)
  { assert(0); return {}; }

  void prefetch(size_t numElements) const;

  // Arithmetic operators
  friend multi_ptr& operator++(multi_ptr& mp) { assert(0); return mp; }
  friend multi_ptr operator++(multi_ptr& mp, int)
  { assert(0); return mp; }
  friend multi_ptr& operator--(multi_ptr& mp)
  { assert(0); return mp; }
  friend multi_ptr operator--(multi_ptr& mp, int)
  { assert(0); return mp; }
  friend multi_ptr& operator+=(multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr& operator-=(multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr operator+(const multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr operator-(const multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend reference operator*(const multi_ptr& lhs)
  { assert(0); return lhs; }

  friend bool operator==(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }

  friend bool operator==(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }

private:
  pointer m_p;
};

// Specialization of multi_ptr for void and const void
// VoidType can be either void or const void
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<const void, Space, DecorateAddress> { /* todo */ };

template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<void, Space, DecorateAddress>
{
  using VoidType = void;
public:

  static constexpr bool is_decorated = DecorateAddress==access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = VoidType;
  using __unspecified__ = value_type; // todo
  using pointer = std::conditional_t<is_decorated, __unspecified__ *,
                                     std::add_pointer_t<value_type>>;
  using difference_type = std::ptrdiff_t;
  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr();
  multi_ptr(const multi_ptr&);
  multi_ptr(multi_ptr&&);
  explicit multi_ptr(
    typename multi_ptr<VoidType, Space, access::decorated::yes>::pointer);
  multi_ptr(std::nullptr_t);

  // Only if Space == global_space
  template <typename ElementType, int dims, access_mode Mode,
            access::placeholder isPlaceholder>
  multi_ptr(accessor<ElementType, dims,Mode, target::device, isPlaceholder>);

  // Only if Space == local_space
  //template <typename ElementType, int dims>
  //multi_ptr(local_accessor<ElementType, dims>);
  // Deprecated
  // Only if Space == local_space
  template <typename ElementType, int dims, access_mode Mode,
            access::placeholder isPlaceholder>
  multi_ptr(accessor<ElementType,dims,Mode,target::local,isPlaceholder>);

  // Assignment operators
  multi_ptr &operator=(const multi_ptr&);
  multi_ptr &operator=(multi_ptr&&);
  multi_ptr &operator=(std::nullptr_t);

  pointer get() const;

  // Conversion to the underlying pointer type
  explicit operator pointer() const;

  // Explicit conversion to a multi_ptr<ElementType>
  // If VoidType is const, ElementType must be as well
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const;

  // Implicit conversion to the non-decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::no>() const
    requires(is_decorated)
  { assert(0); return {}; }

  // Implicit conversion to the decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::yes>() const
    requires(!is_decorated)
  { assert(0); return {}; }

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space, DecorateAddress>() const
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }

  friend bool operator>=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator==(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }

private:
  pointer m_p;
};

// Deprecated, address_space_cast should be used instead.
template <
  typename ElementType,
  access::address_space Space,
  access::decorated DecorateAddress
>
multi_ptr<ElementType, Space, DecorateAddress>
make_ptr(ElementType *p) { return {p}; }

template <
  access::address_space Space,
  access::decorated DecorateAddress,
  typename ElementType
>
multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *);

// Deduction guides
// Does the specification indicate what the 3rd template parameter for multi_ptr
// should be here?
/*
template <
  int dimensions,
  access_mode Mode,
  access::placeholder isPlaceholder,
  class T
>
multi_ptr(accessor<T, dimensions, Mode, target::device, isPlaceholder>)
  -> multi_ptr<T, access::address_space::global_space>;

template <
  int dimensions,
  access_mode Mode,
  access::placeholder isPlaceholder,
  class T
>
multi_ptr(local_accessor<T, dimensions>)
  -> multi_ptr<T, access::address_space::local_space>;
*/

template <typename ElementType, access::address_space Space>
class [[deprecated]] multi_ptr<ElementType, Space, access::decorated::legacy>
{
public:
  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions.
  using pointer_t =
    typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
    typename multi_ptr<const ElementType, Space, access::decorated::yes>
      ::pointer;
  using reference_t =
    typename multi_ptr<ElementType, Space, access::decorated::yes>::reference;
  using const_reference_t =
    typename multi_ptr<const ElementType, Space, access::decorated::yes>
      ::reference;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr();
  multi_ptr(const multi_ptr& p) : m_p{p.m_p} {}
  multi_ptr(multi_ptr&&);
  // multi_ptr(pointer_t); // this can have the same signature as the next:
  multi_ptr(ElementType* p) : m_p{p} {}
  multi_ptr(std::nullptr_t p) : m_p{p} {}
  ~multi_ptr() {}

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr&);
  multi_ptr &operator=(multi_ptr&&);
  //multi_ptr &operator=(pointer_t); // as above - possible signature clash
  multi_ptr &operator=(ElementType* p) { m_p = p; return *this; }
  multi_ptr &operator=(std::nullptr_t p) { m_p = p; return *this; }
  friend ElementType& operator*(const multi_ptr& mp) { return *mp.m_p; }
  ElementType* operator->() const { return m_p; }

  // Only if Space == global_space
  template <int dims, access_mode Mode, access::placeholder isPlaceholder>
  multi_ptr(accessor<ElementType, dims, Mode, target::device, isPlaceholder>);

  // Only if Space == local_space
  template <int dims, access_mode Mode, access::placeholder isPlaceholder>
  multi_ptr(accessor<ElementType, dims, Mode, target::local, isPlaceholder>);

  // Only if Space == constant_space
  template <int dims, access_mode Mode, access::placeholder isPlaceholder>
  multi_ptr(accessor<ElementType, dims, Mode, target::constant_buffer, isPlaceholder>);

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_p; }

  // Implicit conversion to the underlying pointer type
  operator ElementType*() const;

  // Implicit conversion to a multi_ptr<void>
  operator multi_ptr<void, Space, access::decorated::legacy>() const
    requires(!std::is_const_v<element_type>)
  { assert(0); return {}; }

  // Implicit conversion to a multi_ptr<const void>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const
    requires(std::is_const_v<element_type>)
  { assert(0); return {}; }

  // Implicit conversion to multi_ptr<const ElementType, Space>
  operator multi_ptr<const ElementType,Space,access::decorated::legacy>() const;

  // Arithmetic operators
  friend multi_ptr& operator++(multi_ptr& mp) { assert(0); return mp; }
  friend multi_ptr operator++(multi_ptr& mp, int) { assert(0); return mp; }
  friend multi_ptr& operator--(multi_ptr& mp) { assert(0); return mp; };

  friend multi_ptr operator--(multi_ptr& mp, int)
  { assert(0); return mp; }
  friend multi_ptr& operator+=(multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr& operator-=(multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr operator+(const multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }
  friend multi_ptr operator-(const multi_ptr& lhs, difference_type r)
  { assert(0); return lhs; }

  void prefetch(size_t numElements) const { assert(0); }

  friend bool operator==(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }

  friend bool operator==(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }

private:
  pointer_t m_p;
};

// Legacy interface, inherited from 1.2.1.
// Deprecated.
// Specialization of multi_ptr for void and const void
// VoidType can be either void or const void
template <access::address_space Space>
class [[deprecated]] multi_ptr<const void, Space, access::decorated::legacy> {};

template <access::address_space Space>
class [[deprecated]] multi_ptr<void, Space, access::decorated::legacy>
{
  using VoidType = void;

public:
  using element_type = VoidType;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t =
    typename multi_ptr<VoidType, Space, access::decorated::yes>::pointer;
  //using const_pointer_t =
  //typename multi_ptr<const VoidType, Space, access::decorated::yes>::pointer;

  static constexpr access::address_space address_space = Space;
  // Constructors
  multi_ptr();
  multi_ptr(const multi_ptr&);
  multi_ptr(multi_ptr&&);
  //multi_ptr(pointer_t); // see discussion for previous specialisation
  multi_ptr(VoidType*);
  multi_ptr(std::nullptr_t);
  ~multi_ptr();

  // Assignment operators
  multi_ptr &operator=(const multi_ptr&);
  multi_ptr &operator=(multi_ptr&&);
  //multi_ptr &operator=(pointer_t); // as above
  multi_ptr &operator=(VoidType*);
  multi_ptr &operator=(std::nullptr_t);

  // Only if Space == global_space
  template <typename ElementType, int dims, access_mode Mode>
  multi_ptr(accessor<ElementType, dims, Mode, target::device>);

  // Only if Space == local_space
  template <typename ElementType, int dims, access_mode Mode>
  multi_ptr(accessor<ElementType, dims, Mode, target::local>);

  // Only if Space == constant_space
  template <typename ElementType, int dims, access_mode Mode>
  multi_ptr(accessor<ElementType, dims, Mode, target::constant_buffer>);

  // Returns the underlying OpenCL C pointer
  pointer_t get() const;

  // Implicit conversion to the underlying pointer type
  operator VoidType*() const;

  // Explicit conversion to a multi_ptr<ElementType>
  // If VoidType is const, ElementType must be as well
  template <typename ElementType>
  explicit
  operator multi_ptr<ElementType, Space, access::decorated::legacy>() const;

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, const multi_ptr& rhs)
  { assert(0); return {}; }

  friend bool operator==(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator!=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator<=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }
  friend bool operator>=(const multi_ptr& lhs, std::nullptr_t)
  { assert(0); return {}; }

  friend bool operator==(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator!=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator<=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
  friend bool operator>=(std::nullptr_t, const multi_ptr& rhs)
  { assert(0); return {}; }
};

// Section 4.7.7.2. Explicit pointer aliases
// Template specialization aliases for different pointer address spaces
template <
  typename ElementType,
  access::decorated IsDecorated = access::decorated::legacy
>
using global_ptr =
  multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template <
  typename ElementType,
  access::decorated IsDecorated = access::decorated::legacy
>
using local_ptr =
  multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

// Deprecated in SYCL 2020
template <typename ElementType>
using constant_ptr =
  multi_ptr<ElementType, access::address_space::constant_space,
            access::decorated::legacy>;

template <
  typename ElementType,
  access::decorated IsDecorated = access::decorated::legacy
>
using private_ptr =
  multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes non-decorated pointer while keeping the
// address space information internally.

template <typename ElementType>
using raw_global_ptr =
  multi_ptr<ElementType, access::address_space::global_space,
            access::decorated::no>;

template <typename ElementType>
using raw_local_ptr =
  multi_ptr<ElementType, access::address_space::local_space,
            access::decorated::no>;

template <typename ElementType>
using raw_private_ptr =
  multi_ptr<ElementType, access::address_space::private_space,
           access::decorated::no>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes decorated pointer.
template <typename ElementType>
using decorated_global_ptr =
  multi_ptr<ElementType, access::address_space::global_space,
            access::decorated::yes>;

template <typename ElementType>
using decorated_local_ptr =
  multi_ptr<ElementType, access::address_space::local_space,
            access::decorated::yes>;

template <typename ElementType>
using decorated_private_ptr =
  multi_ptr<ElementType, access::address_space::private_space,
            access::decorated::yes>;

// Section 4.9.4 Command group handler class
class handler
{
  queue& q_;
  context& context_;
  std::vector<event*> buffer_events_;
  event kernel_event_;

  friend class queue; // ctor
  template <typename, int, access_mode, target, access::placeholder>
  friend class accessor; // q_
  handler(queue& q, context& c) : q_{q}, context_{c} {}

  template <int dims, typename K>
  void parallel_for(range<dims>, const K& k, const item<dims>);

  template <int dims, typename K>
  void parallel_for(range<dims>, const K& k, const id<dims>);

public:

  template <typename K>
  void single_task(const K&);

  // should we also consider that k may use an item<dims,false> ?
  template <int dims, typename K>
  auto parallel_for(range<dims> r, const K& k) {
    using namespace std;
    using t1 = conditional_t<is_invocable_v<K,id<dims>>,id<dims>,item<dims>>;
    using t2 = conditional_t<detail::is_generic_v<K>, item<dims>, t1>;
    parallel_for(r,k,t2{});
  }

  template <int dims, typename K>
  void parallel_for(nd_range<dims>, const K&);

  // Deprecated in SYCL 2020
  template <int dims, typename K>
  void parallel_for(range<dims>, id<dims>, const K&);
};

// Section 4.6.5 Queue interface
// All constructors will implicitly construct a SYCL platform, device
// and context in order to facilitate the construction of the queue.
class queue
{
  device dev_;
  context& context_{detail::g_context};

public:
  friend class handler; // handler access to q_

  explicit queue(const property_list &ps = {}) { }

  explicit queue(const async_handler &ah, const property_list &ps = {}) { }

  template <typename DeviceSelector>
  explicit queue(const DeviceSelector &sel,
                 const property_list &ps = {}) { }

  template <typename DeviceSelector>
  explicit queue(const DeviceSelector &sel,
                 const async_handler &asyncHandler,
                 const property_list &ps = {}) { }

  explicit queue(const device &syclDevice, const property_list &propList = {})
  { assert(0); }

  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {}) { assert(0); }

  template <typename DeviceSelector>
  explicit queue(const context &c, const DeviceSelector &sel,
                 const property_list &ps = {}) { assert(0); }

  template <typename DeviceSelector>
  explicit queue(const context &c, const DeviceSelector &sel,
                 const async_handler &asyncHandler,
                 const property_list &ps = {}) { assert(0); }

  explicit queue(const context &c, const device &syclDevice,
                 const property_list &ps = {}) { assert(0); }

  explicit queue(const context &c, const device &syclDevice,
                 const async_handler &asyncHandler,
                 const property_list &ps = {}) { assert(0); }

  ~queue() { wait(); }

  backend get_backend() const noexcept { assert(0); return {}; }
  context get_context() const { return context_; }
  device get_device() const { return dev_; }
  bool is_in_order() const { assert(0); return {}; }

  /* -- convenience shortcuts -- */

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(range<dims> r, Rest&&... rest)
  { h_.parallel_for(r, std::forward<Rest>(rest)...); return {}; }

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(range<dims> r, event depEvent, Rest&&... rest)
  { assert(0); return {}; }

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(range<dims> r,
                     const std::vector<event> &depEvents, Rest&&... rest)
  { assert(0); return {}; }

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(nd_range<dims> ndr, Rest&&... rest)
  { assert(0); return {}; }

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(nd_range<dims> ndr, event depEvent, Rest&&... rest)
  { assert(0); return {}; }

  // Parameter pack acts as-if:
  // Reductions&&... reductions, const KernelType &kernelFunc
  template </* typename KernelName, */ int dims, typename... Rest>
  event parallel_for(nd_range<dims> ndr,
                     const std::vector<event> &depEvents, Rest&&... rest)
  { assert(0); return {}; }

  /* -- USM functions -- */

  event memcpy(void* dest, const void* src, size_t numBytes) {
#ifdef __NVCOMPILER
    cudaMemcpy(dest, src, numBytes, cudaMemcpyDefault);
#else
    std::memcpy(dest, src, numBytes);
#endif
    return {};
  }

  event memcpy(void* dest, const void* src, size_t numBytes, event depEvent) {
    assert(0);
    return {};
  }

  event memcpy(void* dest, const void* src, size_t numBytes,
               const std::vector<event> &depEvents) {
    assert(0);
    return {};
  }

  template <typename param>
  typename param::return_type get_info() const
  {
    if constexpr (std::is_same_v<param,sycl::info::queue::context>) {
      return get_context();
    } else if constexpr (std::is_same_v<param,sycl::info::queue::device>) {
      return get_device();
    }

    assert(0);
    return {};
  }

  void wait() { cudaStreamSynchronize(0); }

  template <typename T>
  event submit(T cgf)
  {
    handler h{*this, context_};
    cgf(h);
    std::cout << h.buffer_events_.size() << " accessors considered!\n";
    for (event* p: h.buffer_events_)
      *p = h.kernel_event_;
    return h.kernel_event_;
  }
};

namespace detail {

template <typename T, T ...xs>
using is = std::integer_sequence<T,xs...>;

template <typename T, T x, T... xs>
auto make_stop(const size_t r0, is<T,x,xs...>) {
  return id<1+sizeof...(xs)>{r0,(xs*0)...};
}

template <typename T, T x, T... xs>
auto make_stop(const size_t r0, is<T,x,xs...>, const id<1+sizeof...(xs)> &o) {
  return id<1+sizeof...(xs)>{r0+o[0],o[xs]...};
}

template <typename U, typename A, typename T, T... Is>
constexpr U repack(const A& x, is<T,Is...>) { return U{x[Is]...}; }

template <int dims, typename K>
__global__ void cuda_kernel_launch(const K k)
{
  k(mk_nd_item(id<dims>{})); // The id<dims>{} parameter sets the offset to zero
}

template <typename K>
__global__ void cuda_kernel_launch_single_task(const K k)
{
  k();
}

} // namespace detail

template <typename K>
void handler::single_task(const K& k) {
  detail::cuda_kernel_launch_single_task<K><<<1,1>>>(k);
}

template <int dims, typename K>
void handler::parallel_for(range<dims> r, const K &k, const id<dims>)
{
  auto f = [=]() {
    id<dims> extent{r};
    id stop{detail::make_stop(r[0],std::make_index_sequence<dims>{})};
    detail::iterq begin{id<dims>{},extent}, end{stop,extent};
    std::for_each(detail::g_pol, begin, end, k);
  };
  q_.stdq_.push(f);
}

template <int dims, typename K>
void handler::parallel_for(range<dims> r, const K &k, const item<dims>)
{
  auto f = [=]() {
    using item_t = item<dims,false>;
    id stop{detail::make_stop(r[0],std::make_index_sequence<dims>{})};
    detail::iterq begin{item_t{id<dims>{},r}}, end{item_t{stop,r}};
    std::for_each(detail::g_pol, begin, end, k);
  };
  q_.stdq_.push(f);
}

#ifdef __NVCOMPILER

template <int dims, typename K>
void handler::parallel_for(nd_range<dims> r, const K& k)
{
  static const auto is = std::make_index_sequence<dims>{};
  const dim3 nblocks  = detail::repack<dim3>(r.get_group_range(), is);
  const dim3 nthreads = detail::repack<dim3>(r.get_local_range(), is);
  const dim3 global   = detail::repack<dim3>(r.get_global_range(), is);

  detail::cuda_kernel_launch<dims,K><<<nblocks,nthreads>>>(k);
}

#else

template <int dims, typename K>
void handler::parallel_for(nd_range<dims> r, const K& k)
{
  auto f = [=]() {
    using item_t = nd_item<dims>;
    id stop{detail::make_stop(r.get_global_range()[0],std::make_index_sequence<dims>{})};
    detail::iterq begin{item_t{id<dims>{},r}}, end{item_t{stop,r}};
    std::for_each(detail::g_pol, begin, end, k);
  };
  q_.stdq_.push(f);
}

#endif

// Deprecated in SYCL 2020
template <int dims, typename K>
void handler::parallel_for(range<dims> r, id<dims> o, const K &k)
{
  auto f = [=]() {
    using item_t = item<dims,true>;
    id stop{detail::make_stop(r[0],std::make_index_sequence<dims>{},o)};
    detail::iterq begin{item_t{o,r,o}}, end{item_t{stop,r,o}};
    std::for_each(detail::g_pol, begin, end, k);
  };
  q_.stdq_.push(f);
}

// Does parallel_for with offset support using id?

// Section 4.7.2.2 Buffer properties
namespace property {
namespace buffer {

class use_host_ptr
{
public:
  use_host_ptr() = default;
};

class use_mutex
{
public:
  use_mutex(std::mutex &mutexRef) : mutexRef_{mutexRef} {}
  std::mutex* get_mutex_ptr() const { return &mutexRef_; }

private:
  std::mutex& mutexRef_;
};

class context_bound
{
public:
  context_bound(context boundContext) : boundContext_{boundContext} {}
  context get_context() const { return boundContext_; }

private:
  context boundContext_;
};

} // namespace buffer
} // namespace property

namespace detail {

template <typename C, typename T>
concept is_contiguous = requires(C c) {
  std::size(c);
  requires std::is_convertible_v<std::remove_pointer_t<decltype(std::data(c))> (*)[], const T (*)[]>;
};

inline bool is_aligned(const void * ptr, std::uintptr_t alignment) noexcept {
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignment);
};

} // namespace detail

template <typename T, int dims, typename AllocT>
requires(dims==1||dims==2||dims==3)
class buffer
{
public:

  template <typename, int, access_mode, target, access::placeholder>
  friend class accessor;
  template <typename, int, access_mode>
  friend class host_accessor;

  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using allocator_type = AllocT;

  buffer(const range<dims> &r, const property_list &ps = {})
    : range_{r},
      h_data_{alloc_.allocate(r.size()),
              [this](auto* p){ alloc_.deallocate(p, range_.size()); }} {}

  buffer(const range<dims> &r, AllocT alloc, const property_list &ps = {})
  { assert(0); }

  buffer(T* hostData, const range<dims>& r, const property_list &ps = {})
    : range_{r}, h_data_{hostData,[](auto){}}, ps_{ps}
  {
    const bool well_aligned = detail::is_aligned(hostData, alignof(value_type));
    const bool use_host_ptr = false;
    if (!well_aligned && !use_host_ptr) {
      std::cerr << "Warning: misaligned host data passed to buffer.\n";
      h_data_.reset(alloc_.allocate(size()),
                    [this](auto* p){ alloc_.deallocate(p, size()); });
      std::copy_n(hostData, size(), h_data_.get());
      h_user_data_ = hostData;
    }
  }

  buffer(T *hostData, const range<dims> &r,
         AllocT alloc, const property_list &ps = {})
  { assert(0); }

  buffer(const T *hostData, const range<dims> &r, const property_list &ps = {})
  { assert(0); }

  buffer(const T *hostData, const range<dims> &r,
         AllocT alloc, const property_list &ps = {})
  { assert(0); }

  template <typename Container>
  buffer(Container &container, AllocT alloc, const property_list &ps = {})
  requires(detail::is_contiguous<Container,T> && dims==1)
    : buffer{container.data(), range<dims>{container.size()}, alloc, ps} { }

  template <typename Container>
  buffer(Container &container, const property_list &ps = {})
  requires(detail::is_contiguous<Container,T> && dims==1)
    : buffer{container.data(), range<dims>{container.size()}, ps} { }

  buffer(const std::shared_ptr<T> &hostData, const range<dims> &r,
         AllocT alloc, const property_list &ps = {})
  { assert(0); }

  buffer(const std::shared_ptr<T> &hostData, const range<dims> &r,
         const property_list &ps = {})
  { assert(0); }

  buffer(const std::shared_ptr<T[]> &hostData, const range<dims> &r,
         AllocT alloc, const property_list &ps = {})
  { assert(0); }

  buffer(const std::shared_ptr<T[]> &hostData, const range<dims> &r,
         const property_list &ps = {})
  { assert(0); }

  // Uses buffer<T, 1> in the spec. Available only when: (dims == 1)?
  template <class InputIterator>
  buffer(InputIterator first, InputIterator last, AllocT alloc,
         const property_list &ps = {})
    requires(dims==1)
  { assert(0); }

  // Uses buffer<T, 1> in the spec. Available only when: (dims == 1)?
  template <class InputIterator>
  buffer(InputIterator first, InputIterator last, const property_list &ps = {})
    requires(dims==1)
  { assert(0); }

  buffer(buffer& b, const id<dims> &baseIndex, const range<dims> &subRange)
  { assert(0); }

  /* -- common reference semantics -- */

  buffer(const buffer&)            = default;
  buffer(buffer&&)                 = default;
  buffer& operator=(const buffer&) = default;
  buffer& operator=(buffer&&)      = default;

private:
  void wait_and_copy_back_data()
  {
    event_.wait();
    cudaMemcpy(h_data_.get(), d_data_, byte_size(), cudaMemcpyDeviceToHost);
    if (h_user_data_) {
      std::copy_n(h_data_.get(), size(), h_user_data_);
    }
  }

public:
  ~buffer() { wait_and_copy_back_data(); }

  // property interface members

  template <typename Property>
  bool has_property() const noexcept
  {
    for (const auto& v : pl_.ps_)
      if (holds_alternative<Property>(v))
        return true;
    return false;
  }

  template <typename Property>
  Property get_property() const
  {
    for (const auto& v : pl_.ps_)
      if (holds_alternative<Property>(v))
        return std::get<Property>(v);
    throw sycl::exception{errc::invalid,
                          "the buffer was not constructed with this property."};
  }

  range<dims> get_range() const { return range_; }

  size_t byte_size() const noexcept { return size() * sizeof(value_type); }

  size_t size() const noexcept { return range_.size(); }

  [[deprecated]]  size_t get_count() const { return size(); }

  [[deprecated]] size_t get_size() const { return byte_size(); }

  AllocT get_allocator() const { return alloc_; }

  // Table 4.32/4.33

  // Returns a valid accessor to the buffer with the specified access mode and
  // target in the command group buffer. The value of target can be
  // target::device or target::constant_buffer.

  template <
    access_mode Mode = access_mode::read_write,
    target Targ = target::device
  >
  accessor<T, dims, Mode, Targ> get_access(handler &cgh)
  {
    return {*this, cgh};
  }

  // Deprecated in SYCL 2020. Use get_host_access() instead
  // Returns a valid host accessor to the buffer with the specified access mode
  // and target.

  template <access_mode Mode>
  [[deprecated]]
  accessor<T, dims, Mode, target::host_buffer>
  get_access() { assert(0); return {}; }

  // Returns a valid accessor to the buffer with the specified access mode and
  // target in the command group buffer. Only the values starting from the
  // given offset and up to the given range are guaranteed to be updated. The
  // value of target can be target::device or target::constant_buffer.

  template <
    access_mode Mode = access_mode::read_write,
    target Targ = target::device
  >
  accessor<T, dims, Mode, Targ>
  get_access(handler &cgh, range<dims> accessRange, id<dims> accessOffset = {})
  { assert(0); return {}; }

  // Deprecated in SYCL 2020. Use get_host_access() instead
  // Returns a valid host accessor to the buffer with the specified access mode
  // and target.  Only the values starting from the given offset and up to
  // the given range are guaranteed to be updated. The value of target can only
  // be target::host_buffer .

  template <access_mode Mode>
  [[deprecated]]
  accessor<T, dims, Mode, target::host_buffer>
  get_access(range<dims> accessRange, id<dims> accessOffset = {})
  {
    assert(0); return {};
  }

  // Returns a valid accessor as if constructed via passing the buffer and all
  // provided arguments to the SYCL accessor.
  // Possible implementation: return accessor { *this, args... };

  template <typename... Ts>
  auto get_access(Ts... args) { return accessor{ *this, args...}; };

  template<typename... Ts>
  auto get_host_access(Ts... args) { return host_accessor{*this, args...}; }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr)
  { assert(0); }

  void set_write_back(bool flag = true)
  { assert(0); }

  bool is_sub_buffer() const
  { assert(0); return {}; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<
    ReinterpretT, ReinterpretDim,
    typename std::allocator_traits<AllocT>::template rebind_alloc<ReinterpretT>
  >
  reinterpret(range<ReinterpretDim> reinterpretRange) const
  { assert(0); return {}; }

  template <typename ReinterpretT, int ReinterpretDim = dims>
  buffer<
    ReinterpretT, ReinterpretDim,
    typename std::allocator_traits<AllocT>::template rebind_alloc<ReinterpretT>
  >
  reinterpret() const
  requires( ReinterpretDim==1 ||
           (ReinterpretDim==dims && sizeof(ReinterpretT)==sizeof(T)))
  { assert(0); return {}; }

private:

  allocator_type alloc_{};
  const range<dims> range_{};
  queue* pq_{};
  std::shared_ptr<T[]> h_data_{};
  T* h_user_data_{};
  T* d_data_{}; // needed? Use the context's allocations_
  buffer* original_{this};
  event event_{}; // needed?
  property_list ps_{};
};

// Deduction guides
template <class InputIterator, class AllocT>
buffer(InputIterator, InputIterator, AllocT, const property_list & = {})
 -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1, AllocT>;

template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
 -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template <class T, int dims, class AllocT>
buffer(const T *, const range<dims> &, AllocT, const property_list & = {})
 -> buffer<T, dims, AllocT>;

template <class T, int dims>
buffer(const T *, const range<dims> &, const property_list & = {})
 -> buffer<T, dims>;

template <class Container, class AllocT>
buffer(Container &, AllocT, const property_list & = {})
  -> buffer<typename Container::value_type, 1, AllocT>;

template <class Container>
buffer(Container &, const property_list & = {})
  -> buffer<typename Container::value_type, 1>;

namespace detail {

  size_t linear_offset(const id<1> &o, const range<1> &r) { return o[0]; }
  size_t linear_offset(const id<2> &o, const range<2> &r)
  { return (o[0] * r[1]) + o[1]; }
  size_t linear_offset(const id<3> &o, const range<3> &r)
  { return (o[0] * r[1] * r[2]) + (o[1] * r[2]) + o[2]; }

} // namespace detail

namespace detail
{

template <typename, int, access_mode>
struct indexer;

template <typename dataT, access_mode accessmode>
struct indexer<dataT, 1, accessmode>
{
  dataT &operator[](size_t index) const { return data_[index]; }

#ifdef __NVCOMPILER
  dataT* data_{};
#else
  std::shared_ptr<dataT[]> data_;
#endif
  const range<1> range_;
};

template <typename dataT, access_mode accessmode>
struct indexer<dataT, 2, accessmode>
{
  indexer<dataT, 1, accessmode> operator[](size_t index) const {
#ifdef __NVCOMPILER
    return {data_ + range_[1] * index, {range_[1]}};
#else
    return {{data_, data_.get() + range_[1] * index}, {range_[1]}};
#endif
  }

#ifdef __NVCOMPILER
  dataT* data_{};
#else
  std::shared_ptr<dataT[]> data_;
#endif
  const range<2> range_;
};

} // namespace detail

// Section 4.7.6.9.1 Device buffer accessor interface
template <
  typename dataT,
  int dims,
  access_mode AccessMode,
  access::target AccessTarget,
  access::placeholder isPlaceholder
>
class accessor
{
public:

  using value_type =
    std::conditional_t<AccessMode == access_mode::read, const dataT, dataT>;
  using reference = value_type&;
  using const_reference = const dataT&;
  //template <access::decorated IsDecorated>
  //using accessor_ptr =   // multi_ptr to value_type with target address space,
  //  __pointer_class__;   // unspecified for access_mode::host_task
  //using iterator = __unspecified_iterator__<value_type>;
  //using const_iterator = __unspecified_iterator__<const value_type>;
  //using reverse_iterator = std::reverse_iterator<iterator>;
  //using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  //using difference_type = typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  accessor() = default;

  template <typename AllocT>
  accessor(buffer<dataT, 1, AllocT> &buf, const property_list &ps = {})
    requires(dims==0)
  { assert(0); }

  template <typename AllocT>
  accessor(buffer<dataT, 1, AllocT> &buf,
           handler &cgh, const property_list &ps = {})
    requires(dims==0)
  { assert(0); }

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, TagT tag,
           const property_list &ps = {})
    requires(dims>0) : accessor{buf, ps} {}

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh,
           const property_list &ps = {})
    requires(dims>0)
    : range_{buf.range_}, offset_{}
  {
    auto& allocs = cgh.context_.allocations_;
//    if (allocs.find(buf.original_) == allocs.end())
    if (allocs.contains(buf.original_))
    {
      std::cerr << "Allocating device memory.\n";
      cudaMalloc(&buf.d_data_, byte_size());
      cudaMemcpy( buf.d_data_, buf.h_data_.get(), byte_size(),
                  cudaMemcpyHostToDevice);
      detail::device_allocation a{buf.d_data_};
      allocs[buf.original_] = std::move(a);
    }

    // See Table 57
    if constexpr (AccessTarget==target::device)
    {
      if constexpr (AccessMode==access_mode::read_write ||
                    AccessMode==access_mode::write)
      {
        cgh.buffer_events_.push_back(&buf.event_);   // used by queue::submit
      }
    }

    buf.pq_ = &cgh.q_;
    d_data_ = buf.d_data_;
  }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh, TagT tag,
           const property_list &ps = {})
    requires(dims>0) : accessor{buf, cgh, ps} {}

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf,
           range<dims> accessRange, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange, TagT tag,
           const property_list &ps = {})
    requires(dims>0) : accessor{buf, accessRange, ps} {}

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange,
           id<dims> accessOffset, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange,
           id<dims> accessOffset, TagT tag, const property_list &ps = {})
    requires(dims>0) : accessor{buf, accessRange, accessOffset, ps} {}

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh,
           range<dims> accessRange, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh,
           range<dims> accessRange, TagT tag, const property_list &ps = {})
    requires(dims>0) : accessor{buf, cgh, accessRange, ps} {}

  template <typename AllocT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh,
           range<dims> accessRange, id<dims> accessOffset,
           const property_list &ps = {})
    requires(dims>0)
    : data_{buf.data_}, range_{accessRange}, offset_{accessOffset}
  { assert(0); buf.pq_ = &cgh.q_; }

  template <typename AllocT, typename TagT>
  accessor(buffer<dataT, dims, AllocT> &buf, handler &cgh,
           range<dims> accessRange, id<dims> accessOffset, TagT tag,
           const property_list &ps = {})
    requires(dims>0) : accessor{buf, cgh, accessRange, accessOffset, ps} {}

  void swap(accessor &other) { assert(0); }

  bool is_placeholder() const { return isPlaceholder == placeholder::true_t; }
  size_type byte_size() const noexcept { return range_.size() * sizeof(dataT); }
  size_type size() const noexcept { return range_.size(); }
  size_type max_size() const noexcept { assert(0); return {}; }

  [[deprecated]] size_t get_size() const
    requires(AccessTarget==target::device) { return byte_size(); }
  [[deprecated]] size_t get_count() const
    requires(AccessTarget==target::device) { return size(); }

  bool empty() const noexcept { return size() == 0; }

  range<dims> get_range() const requires(dims>0) { return range_; }

  id<dims> get_offset() const requires(dims>0) { return offset_; }

  operator reference() const requires(dims==0) { return (*this)[0]; }

#if 0
  // todo: review these operator[] signatures
  template <int D>
  reference operator[](id<D> i) const
  { return data_[detail::linear_offset(i+offset_,range_)]; }

  // These id types must have the same dims as the accessor
  template <int D, bool O>
  reference operator[](item<D, O> i) const { return (*this)[i.get_id()]; }

  // Available only when: (accessMode != access_mode::atomic && dimensions == 1)
  reference operator[](size_t index) const { return data_[index]; }

  // accessMode == access_mode::read
  /*const_reference operator[](id<dims> index) const
  {
  }*/
#else
  /* Available only when: (dims > 0) */
  template <int D>
  reference operator[](id<D> i) const
  { return d_data_[detail::linear_offset(i+offset_,range_)]; }

  template <int d = dims>
  std::enable_if_t<(d==1), reference>
  operator[](size_t i) const {
    return d_data_[i+offset_[0]];
  }

  /* Available only when: dims > 1 */
  // Off-piste: returning an __unspecified__ ... not an __unspecified__&
  template <int d = dims>
  std::enable_if_t<(d==2), const detail::indexer<dataT, dims-1, AccessMode>>
  operator[](size_t index) const {
    return {d_data_ + range_[1] * index, {range_[1]}};
  }

  template <int d = dims>
  std::enable_if_t<(d==3), const detail::indexer<dataT, dims-1, AccessMode>>
  operator[](size_t index) const {
#ifdef __NVCOMPILER
    return {data_ + index * range_[1] * range_[2], {range_[1], range_[2]}};
#else
    return {{data_, data_.get() + index * range_[1] * range_[2]}, {range_[1], range_[2]}};
#endif
  }
#endif

  /* Deprecated: Available only when:
     (accessMode == access_mode::atomic && dimensions == 0) */
  //operator cl::sycl::atomic<dataT,access::address_space::global_space> () const;

  /* Deprecated: Available only when:
     (accessMode == access_mode::atomic && dimensions == 1) */
  //cl::sycl::atomic<dataT, access::address_space::global_space> operator[](
  //  id<dimensions> index) const;

  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return d_data_;
  }

  //template <access::decorated IsDecorated>
  //accessor_ptr<IsDecorated> get_multi_ptr() const noexcept;

  /*
  iterator begin() const noexcept;
  iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;
  reverse_iterator rbegin() const noexcept;
  reverse_iterator rend() const noexcept;
  const_reverse_iterator crbegin() const noexcept;
  const_reverse_iterator crend() const noexcept;
  */

  dataT* d_data_{}; // cannot be a reference as the buffer is stored on the host
  const range<dims> range_;
  const id<dims> offset_;
};

template <
  typename dataT,
  int dims,
  access_mode accessmode
>
class host_accessor
{
public:

  using value_type =
    std::conditional_t<accessmode == access_mode::read, const dataT, dataT>;
  using reference =
    std::conditional_t<accessmode == access_mode::read, const dataT&, dataT&>;
  using const_reference = const dataT&;

#if 0
  using iterator =
  // const dataT* when (accessmode == access::mode::read),
  __pointer_type__; // dataT* otherwise
  using const_iterator = const dataT *;
  using difference_type =
  typename std::iterator_traits<iterator>::difference_type;
#endif
  using size_type = size_t;

  host_accessor() = default;

  template <typename AllocT>
  host_accessor(buffer<dataT, 1, AllocT> &buf, const property_list &ps = {})
    requires(dims==0)
  { assert(0); }

  template <typename AllocT>
  host_accessor(buffer<dataT, dims, AllocT> &buf, const property_list &ps = {})
    requires(dims>0)
    : h_data_{buf.h_data_}, d_data_{buf.d_data_}, range_{buf.range_}
  {
    // if (buf.pq_) buf.pq_->wait();
    buf.wait_and_copy_back_data();
  }

  template <typename AllocT, typename TagT>
  host_accessor(buffer<dataT, dims, AllocT> &buf, TagT tag,
                const property_list &ps = {})
    requires(dims>0)
    : host_accessor{buf, ps} {}

  template <typename AllocT>
  host_accessor(buffer<dataT, dims, AllocT> &buf,
                range<dims> accessRange, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  host_accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange,
                TagT tag, const property_list &ps = {})
    requires(dims>0)
    : host_accessor{buf, accessRange, ps} {}

  template <typename AllocT>
  host_accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange,
                id<dims> accessOffset, const property_list &ps = {})
    requires(dims>0)
  { assert(0); }

  template <typename AllocT, typename TagT>
  host_accessor(buffer<dataT, dims, AllocT> &buf, range<dims> accessRange,
                id<dims> accessOffset, TagT tag, const property_list &ps = {})
    requires(dims>0)
    : host_accessor{buf, accessRange, accessOffset, ps} {}

  ~host_accessor()
  {
    cudaMemcpy(d_data_, h_data_.get(), byte_size(), cudaMemcpyHostToDevice);
  }

  /* -- common interface members -- */

  void swap(host_accessor &other) { assert(0); }
  size_type byte_size() const noexcept { return range_.size() * sizeof(dataT); }
  size_type size() const noexcept { return range_.size(); }
  size_type max_size() const noexcept { assert(0); return {}; }
  bool empty() const noexcept { return size() == 0; }

  range<dims> get_range() const requires(dims>0) { return range_; }
  id<dims> get_offset() const requires(dims>0) { return offset_; }
  operator reference() const requires(dims==0) { return (*this)[0]; }

  reference operator[](id<1> index) const
    requires(dims > 0) { return data_[index[0]]; }
  reference operator[](id<2> index) const
  {
    return h_data_[index[0] * range_[1] + index[1]];
  }

  template <int d = dims>
  requires(d==1)
  reference
  operator[](size_t index) const { return h_data_[index]; }

  /* Available only when: dims > 1 */
  // Off-piste: returning an __unspecified__ ... not an __unspecified__&
  template <int d = dims>
  requires(d==2)
  const detail::indexer<dataT, dims-1, accessmode>
  operator[](size_t index) const {
    return {{h_data_, h_data_.get() + range_[1] * index}, {range_[1]}};
  }

  template <int d = dims>
  requires(d==3)
  const detail::indexer<dataT, dims-1, accessmode>
  operator[](size_t index) const {
    return {{h_data_, h_data_.get() + index * range_[1] * range_[2]},
            {range_[1], range_[2]}};
  }

#if 0
  iterator data() const noexcept { assert(0); return {}; }
  iterator begin() const noexcept { assert(0); return {}; }
  iterator end() const noexcept { assert(0); return {}; }
  const_iterator cbegin() const noexcept { assert(0); return {}; }
  const_iterator cend() const noexcept { assert(0); return {}; }
#endif

  std::shared_ptr<dataT[]> h_data_{};
  void* d_data_{};
  const range<dims> range_{};
  const id<dims> offset_{};
};

// Section 4.8.3.2. Device allocation functions
template <typename T>
T* malloc_device(size_t count, const queue& q, const property_list &ps = {}) {
#ifdef __NVCOMPILER
  T *p;
  cudaMalloc(&p, count * sizeof(T));
  return p;
#else
  // Not C++ new, as sycl::free is untyped
  return static_cast<T*>(std::malloc(count * sizeof(T)));
#endif
}

// Section 4.8.3.3. Host allocation functions
void* malloc_host(size_t nbytes, const queue& q, const property_list &ps = {}) {
  assert(0);
  return {};
}

// Section 4.8.3.4. Shared allocation functions
template <typename T>
T* malloc_shared(size_t count, const queue& q, const property_list& ps = {}) {
  return static_cast<T*>(std::malloc(count * sizeof(T)));
}

// Section 4.8.3.6. Memory deallocation functions
void free(void* ptr, sycl::context& syclContext) { assert(0); }

void free(void* ptr, sycl::queue& q) {
#ifdef __NVCOMPILER
  cudaFree(ptr);
#else
  std::free(ptr);
#endif
}

// Section 4.14.2.1 Vec interface

enum class rounding_mode
{
  automatic,
  rte,
  rtz,
  rtp,
  rtn
};

struct elem
{
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

// Section 4.17.1.
using int2 = vec<int,2>;
using int3 = vec<int,3>;
using int4 = vec<int,4>;
using int8 = vec<int,8>;
using int16 = vec<int,16>;

using float2 = vec<float,2>;
using float3 = vec<float,3>;
using float4 = vec<float,4>;
using float8 = vec<float,8>;
using float16 = vec<float,16>;

using double2 = vec<double,2>;
using double3 = vec<double,3>;
using double4 = vec<double,4>;
using double8 = vec<double,8>;
using double16 = vec<double,16>;

// Section 4.17.1. Description of the built-in types ...

namespace detail {

template <typename T, typename ...Ts>
struct one_of : std::disjunction<std::is_same<T,Ts>...> {};

template <typename T>
struct is_vfloatn : one_of<T,float2,float3,float4,float8,float16> {};

//template <typename T>
//struct mfloatn =

//template <typename T>
//struct marray =

template <typename T>
struct is_floatn : std::disjunction<is_vfloatn<T>> {}; // add mfloat, marray

template <typename T>
struct is_genfloatf : std::disjunction<std::is_same<T,float>,is_floatn<T>> {};

template <typename T>
struct is_vdoublen : one_of<T,double2,double3,double4,double8,double16> {};

template <typename T>
struct is_doublen : std::disjunction<is_vdoublen<T>> {}; // add mdouble, marray

template <typename T>
struct is_genfloatd : std::disjunction<std::is_same<T,double>,is_doublen<T>>{};

// struct is_halfn =
// struct is_genfloath =

template <typename T>              // add is_genfloath
struct is_genfloat : std::disjunction<is_genfloatf<T>,is_genfloatd<T>> {};

template <typename T>
struct is_sgenfloat : one_of<T,float,double> {}; // add half

} // namespace detail

template <typename dataT, int numElements>
class vec
{
public:
  using element_type = dataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = __unspecified__;
#endif

  vec() {}
  explicit vec(const dataT &arg)
    : data_{[&arg]{ decltype(data_) a; a.fill(arg); return a; }()} { }

  // With the spec's signature, this would always be chosen in favour of the
  // above explicit constructor (see no_implicit_conversion in the vectors.cpp)
  //template <typename... argTN>
  //vec(const argTN&... args) : data_{static_cast<dataT>(args)...} { }

  // This signature can avoid the issue mentioned above
  // The implementation still needs to handle vec argument (concatenation etc.)
  template <typename T1, typename T2, typename... Ts>
  vec(const T1& x1, const T2& x2, const Ts&... xs)
    : data_{static_cast<dataT>(x1),
            static_cast<dataT>(x2),
            static_cast<dataT>(xs)...} { }

  vec(const vec<dataT, numElements> &rhs) : data_{rhs.data_} { }

#ifdef __SYCL_DEVICE_ONLY__
  vec(vector_t nativeVector) { assert(0); }
  operator vector_t() const { assert(0); }
#endif

  // Available only when: numElements == 1
  template <int N = numElements>
  operator dataT() const requires(N==1) { return data_[0]; }

  // Section 4.14.2.6.
  static constexpr size_t byte_size() noexcept {
    return sizeof(dataT) * (numElements==3 ? 4 : numElements);
  }
  [[deprecated]] size_t get_size() const { return byte_size(); }

  static constexpr size_t size() noexcept { return numElements; }
  [[deprecated]] size_t get_count() const { return size(); }

  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, numElements> convert() const { assert(0); return {}; }

  template <typename asT>
  asT as() const { assert(0); return {}; }

using __swizzled_vec__ = int; // remove
using RET = int; // remove

  template<int... swizzleIndexes>
  __swizzled_vec__ swizzle() const { assert(0); return {}; }

  // Available only when numElements <= 4.
  // XYZW_ACCESS is: x, y, z, w, subject to numElements.
  //__swizzled_vec__ XYZW_ACCESS() const { assert(0); return {}; }
  element_type x() const { return data_[0]; }
  element_type y() const { return data_[1]; }
  element_type z() const { return data_[2]; }
  element_type w() const { return data_[3]; }

  // Available only numElements == 4.
  // RGBA_ACCESS is: r, g, b, a.
  //__swizzled_vec__ RGBA_ACCESS() const { assert(0); return {}; }
  element_type r() const requires(numElements==4) { return data_[0]; }
  element_type g() const requires(numElements==4) { return data_[1]; }
  element_type b() const requires(numElements==4) { return data_[2]; }
  element_type a() const requires(numElements==4) { return data_[3]; }

  // INDEX_ACCESS is: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD,
  // sE, sF, subject to numElements.
  //__swizzled_vec__ INDEX_ACCESS() const { assert(0); return {}; }
  element_type s0() const { return data_[0]; }
  element_type s1() const { return data_[1]; }
  element_type s2() const { return data_[2]; }
  element_type s3() const { return data_[3]; }
  element_type s4() const { return data_[4]; }
  element_type s5() const { return data_[5]; }
  element_type s6() const { return data_[6]; }
  element_type s7() const { return data_[7]; }
  element_type s8() const { return data_[8]; }
  element_type s9() const { return data_[9]; }
  element_type sA() const { return data_[10]; }
  element_type sB() const { return data_[11]; }
  element_type sC() const { return data_[12]; }
  element_type sD() const { return data_[13]; }
  element_type sE() const { return data_[14]; }
  element_type sF() const { return data_[15]; }

#ifdef SYCL_SIMPLE_SWIZZLES
  // Available only when numElements <= 4.
  // XYZW_SWIZZLE is all permutations with repetition of: x, y, z, w, subject to
  // numElements.
  __swizzled_vec__ XYZW_SWIZZLE() const { assert(0); return {}; }

  // Available only when numElements == 4.
  // RGBA_SWIZZLE is all permutations with repetition of: r, g, b, a.
  __swizzled_vec__ RGBA_SWIZZLE() const { assert(0); return {}; }

#endif // #ifdef SYCL_SIMPLE_SWIZZLES

  // Available only when: numElements > 1.
  __swizzled_vec__ lo() const { assert(0); return {}; }
  __swizzled_vec__ hi() const { assert(0); return {}; }
  __swizzled_vec__ odd() const { assert(0); return {}; }
  __swizzled_vec__ even() const { assert(0); return {}; }

  // load and store member functions
  template <access::address_space addressSpace, access::decorated IsDecorated>
  void load(size_t offset, multi_ptr<const dataT, addressSpace, IsDecorated> ptr)
  { assert(0); }

  template <access::address_space addressSpace, access::decorated IsDecorated>
  void store(size_t offset, multi_ptr<dataT, addressSpace, IsDecorated> ptr) const
  { assert(0); }

  // subscript operator
  dataT &operator[](int index) { return data_[index]; }
  const dataT &operator[](int index) const { return data_[index]; }

  // OP is: +, -, *, /, %
  /* If OP is %, available only when: dataT != float && dataT != double
  && dataT != half. */
  friend vec operator+(const vec &lhs, const vec &rhs) {
    return detail::zip_with<vec>(std::plus{},lhs.data_,rhs.data_);
  }
  friend vec operator+(const vec &lhs, const dataT &rhs) {
    vec v{lhs}; return v += rhs;
  }

  friend vec operator-(const vec &lhs, const vec &rhs) {
    return detail::zip_with<vec>(std::minus{},lhs.data_,rhs.data_);
  }
  friend vec operator-(const vec &lhs, const dataT &rhs) {
    vec v{lhs}; return v -= rhs;
  }

  friend vec operator*(const vec &lhs, const vec &rhs) {
    return detail::zip_with<vec>(std::multiplies{},lhs.data_,rhs.data_);
  }
  friend vec operator*(const vec &lhs, const dataT &rhs) {
    vec v{lhs}; return v *= rhs;
  }

  friend vec operator/(const vec &lhs, const vec &rhs) {
    return detail::zip_with<vec>(std::divides{},lhs.data_,rhs.data_);
  }
  friend vec operator/(const vec &lhs, const dataT &rhs) {
    vec v{lhs}; return v /= rhs;
  }

  friend vec operator%(const vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  {
    return detail::zip_with<vec>(std::modulus{},lhs.data_,rhs.data_);
  }
  friend vec operator%(const vec &lhs, const dataT &rhs) {
    vec v{lhs}; return v %= rhs;
  }

  // OP is: +=, -=, *=, /=, %=
  /* If OP is %=, available only when: dataT != float && dataT != double
  && dataT != half. */
  friend vec &operator+=(vec &lhs, const vec &rhs) {
    for (std::size_t i{0}; i < lhs.size(); ++i) { lhs[i] += rhs[i]; }
    return lhs;
  }
  friend vec &operator+=(vec &lhs, const dataT &rhs) {
    for (auto &x : lhs.data_) { x += rhs; } return lhs;
  }

  friend vec &operator-=(vec &lhs, const vec &rhs) {
    for (std::size_t i{0}; i < lhs.size(); ++i) { lhs[i] -= rhs[i]; }
    return lhs;
  }
  friend vec &operator-=(vec &lhs, const dataT &rhs) {
    for (auto &x : lhs.data_) { x -= rhs; } return lhs;
  }

  friend vec &operator*=(vec &lhs, const vec &rhs) {
    for (std::size_t i{0}; i < lhs.size(); ++i) { lhs[i] *= rhs[i]; }
    return lhs;
  }
  friend vec &operator*=(vec &lhs, const dataT &rhs) {
    for (auto &x : lhs.data_) { x *= rhs; } return lhs;
  }

  friend vec &operator/=(vec &lhs, const vec &rhs) {
    for (std::size_t i{0}; i < lhs.size(); ++i) { lhs[i] /= rhs[i]; }
    return lhs;
  }
  friend vec &operator/=(vec &lhs, const dataT &rhs) {
    for (auto &x : lhs.data_) { x /= rhs; } return lhs;
  }

  friend vec &operator%=(vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  {
    for (std::size_t i{0}; i < lhs.size(); ++i) { lhs[i] %= rhs[i]; }
    return lhs;
  }
  friend vec &operator%=(vec &lhs, const dataT &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  {
    for (auto &x : lhs.data_) { x %= rhs; } return lhs;
  }

  // OP is prefix ++, --
  friend vec &operator++(vec &rhs) { assert(0); return {}; }

  // OP is postfix ++, --
  friend vec operator++(vec& lhs, int) { assert(0); return {}; }

  // OP is unary +, -
  friend vec operator+(vec &rhs) /*const*/ { assert(0); return {}; }

  // OP is: &, |, ^
  /* Available only when: dataT != float && dataT != double && dataT != half. */
  friend vec operator&(const vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec operator&(const vec &lhs, const dataT &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: &=, |=, ^=
  /* Available only when: dataT != float && dataT != double && dataT != half. */
  friend vec &operator&=(vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec &operator&=(vec &lhs, const dataT &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: &&, ||
  friend vec<RET, numElements> operator&&(const vec &lhs, const vec &rhs)
  { assert(0); return {}; }
  friend vec<RET, numElements> operator&&(const vec& lhs, const dataT &rhs)
  { assert(0); return {}; }

  // OP is: <<, >>
  // Available only when: dataT != float && dataT != double && dataT != half
  friend vec operator<<(const vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec operator<<(const vec &lhs, const dataT &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: <<=, >>=
  // Available only when: dataT != float && dataT != double && dataT != half
  friend vec &operator<<=(vec &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec &operator<<=(vec &lhs, const dataT &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: ==, !=, <, >, <=, >=
  friend vec<RET, numElements> operator==(const vec &lhs, const vec &rhs)
  { assert(0); return {}; }
  friend vec<RET, numElements> operator==(const vec &lhs, const dataT &rhs)
  { assert(0); return {}; }

  vec &operator=(const vec<dataT, numElements> &rhs) {
    data_ = rhs.data_; return *this;
  }
  vec &operator=(const dataT &rhs) { data_.fill(rhs); return *this; }

  // Available only when: dataT != float && dataT != double && dataT != half
  friend vec operator~(const vec &v)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec<RET, numElements> operator!(const vec &v)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: +, -, *, /, %
  /* operator% is only available when: dataT != float && dataT != double &&
  dataT != half. */
  friend vec operator+(const dataT &lhs, const vec &rhs){ return vec{lhs}+rhs; }
  friend vec operator-(const dataT &lhs, const vec &rhs){ return vec{lhs}-rhs; }
  friend vec operator*(const dataT &lhs, const vec &rhs){ return vec{lhs}*lhs; }
  friend vec operator/(const dataT &lhs, const vec &rhs){ return vec{lhs}/rhs; }
  friend vec operator%(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value) { return vec{lhs}%rhs; }

  // OP is: &, |, ^
  // Available only when: dataT != float && dataT != double && dataT != half
  friend vec operator&(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec operator|(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec operator^(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: &&, ||
  friend vec<RET, numElements> operatorOP(const dataT &lhs, const vec &rhs)
  { assert(0); return {}; }

  // OP is: <<, >>
  // Available only when: dataT != float && dataT != double && dataT != half
  friend vec operator<<(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }
  friend vec operator>>(const dataT &lhs, const vec &rhs)
    requires(!detail::is_sgenfloat<dataT>::value)
  { assert(0); return {}; }

  // OP is: ==, !=, <, >, <=, >=
  friend vec<RET, numElements> operator==(const dataT &lhs, const vec &rhs)
  { assert(0); return {}; }
private:

  alignas(vec::byte_size()) std::array<dataT, numElements> data_;
};

// Deduction guides
// Available only when: (std::is_same_v<T, U> && ...)
template <class T, class... U>
  requires(std::conjunction_v<std::is_same<T,U>...>)
vec(T, U...) -> vec<T, sizeof...(U) + 1>;

// Section 4.17.5 Math function

inline float sqrt(float x) { return std::sqrt(x); }
inline float sin(float x) { return std::sin(x); }
inline float cos(float x) { return std::cos(x); }
inline float fma(float x, float y, float z) { return std::fma(x,y,z); }
inline float fmin(float x, float y) { return std::fmin(x,y); }
inline float fabs(float x) { return std::fabs(x); }
inline float fmod(float x, float y) { return std::fmod(x,y); }
inline float pow(float x, float y) { return std::pow(x,y); }
inline float tan(float x) { return std::tan(x); }
inline float atan2(float x, float y) { return std::atan2(x,y); }
inline float asin(float x) { return std::asin(x); }
inline float log(float x) { return std::log(x); }

#if 0

template <typename T> inline T acos (T x);
template <typename T> inline T acosh (T x);
template <typename T> inline T acospi (T x);
template <typename T> inline T asin (T x);
template <typename T> inline T asinh (T x);
template <typename T> inline T asinpi (T x);
template <typename T> inline T atan (T y_over_x);
template <typename T> inline T atan2 (T y, T x);
template <typename T> inline T atanh (T x);
template <typename T> inline T atanpi (T x);
template <typename T> inline T atan2pi (T y, T x);
template <typename T> inline T cbrt (T x);
template <typename T> inline T ceil (T x);
template <typename T> inline T copysign (T x, T y);
template <typename T> inline T cos (T x) { return std::cos(x); }
template <typename T> inline T cosh (T x) { return std::cosh(x); }
template <typename T> inline T cospi (T x);
template <typename T> inline T erfc (T x);
template <typename T> inline T erf (T x);
template <typename T> inline T exp (T x);
template <typename T> inline T exp2 (T x);
template <typename T> inline T exp10 (T x);
template <typename T> inline T expm1 (T x);
template <typename T> inline T fabs (T x);
template <typename T> inline T fdim (T x, T y);
template <typename T> inline T floor (T x);
template <typename T> inline T fma (T a,  T b, T c);
template <typename T> inline T fmax (T x, T y);
//template <typename T> inline T fmax (T x, sgenfloat y);
template <typename T> inline T fmin (T x, T y);
//template <typename T> inline T fmin (T x, sgenfloat y);
template <typename T> inline T fmod (T x, T y);
//template <typename T> inline T fract (T x, genfloatptr iptr);
//template <typename T> inline T frexp (T x, genintptr exp);
template <typename T> inline T hypot (T x, T y);
//genint ilogb (T x);
//template <typename T> inline T ldexp (T x, genint k);
template <typename T> inline T ldexp (T x, int k);
template <typename T> inline T lgamma (T x);
//template <typename T> inline T lgamma_r (T x, genintptr signp);
template <typename T> inline T log(T x) { return std::log(x); }
template <typename T> inline T log2(T x) { return std::log2(x); }
template <typename T> inline T log10(T x) { return std::log10(x); }
template <typename T> inline T log1p(T x) { return std::log1p(x); }
template <typename T> inline T logb(T x) { return std::logb(x); }
template <typename T> inline T mad(T a, T b, T c) { return std::fma(a,b,c); }
template <typename T> inline T maxmag (T x, T y)
{
  const T magx{fabs(x)};
  const T magy{fabs(y)};
  return magx > magy ? x : (magy > magx) ? y : fmax(x,y);
}
template <typename T> inline T minmag (T x, T y)
{
  const T magx{fabs(x)};
  const T magy{fabs(y)};
  return magx < magy ? x : (magy < magx) ? y : fmin(x,y);
}
//template <typename T> inline T modf (T x, genfloatptr iptr);
//genfloatf nan (ugenint nancode);
//genfloatd nan (ugenlonginteger nancode);
template <typename T> inline T nextafter(T x, T y) {return std::nextafter(x,y);}
template <typename T> inline T pow (T x, T y) { return std::pow(x,y); }
//template <typename T> inline T pown (T x, genint y);
template <typename T, typename U>
inline std::enable_if_t<std::is_integral_v<U>>
pown(T x, U y) { return std::pow(x,y); }
template <typename T> inline T powr (T x, T y) {
  assert(x>=0); return std::pow(x,y);
}
template <typename T> inline T remainder(T x, T y) {return std::remainder(x,y);}
//template <typename T> inline T remquo (T x, T y, genintptr quo);
template <typename T> inline T remquo(T x, T y, int* quo) {
  return std::remquo(x,y,quo);
}
template <typename T> inline T rint(T x) { return std::rint(x); }
//template <typename T> inline T rootn (T x, genint y);
template <typename T, typename U>
inline std::enable_if_t<std::is_integral_v<U>>
rootn (T x, U y) { return std::pow(x,1/static_cast<T>(y));
}
template <typename T> inline T round(T x) { return std::round(x); }
template <typename T> inline T rsqrt(T x) { return 1/std::sqrt(x); }
template <typename T>
inline std::enable_if_t<detail::is_genfloat<T>::value>
sin(T x) { return std::sin(x); }
//template <typename T> inline T sincos (T x, genfloatptr cosval);
template <typename T> inline T sinh(T x) { return std::sinh(x); }
template <typename T>
inline std::enable_if_t<detail::is_genfloat<T>::value>
sinpi(T x) {
//  return std::sin(std::numbers::pi_v<T> * x); // C++20
  constexpr double pi = 3.14159265358979323846;
  return std::sin(pi * x);
}
template <typename T> inline T sqrt(T x) { return std::sqrt(x); }
template <typename T> inline T tan(T x) { return std::tan(x); }
template <typename T> inline T tanh(T x) { return std::tanh(x); }
template <typename T>
inline std::enable_if_t<detail::is_genfloat<T>::value>
tanpi(T x) {
//  return std::tan(std::numbers::pi_v<T> * x); // C++20
  constexpr double pi = 3.14159265358979323846;
  return std::tan(pi * x);
}
template <typename T>
inline std::enable_if_t<detail::is_genfloat<T>::value>
tgamma(T x) { return std::tgamma(x); }
template <typename T>
inline std::enable_if_t<detail::is_genfloat<T>::value>
trunc (T x) { return std::trunc(x); }
#endif

// Section 4.17.8. Geometric functions

//float4 cross(float4 p0, float4 p1) { assert(0); return {}; }
float3 cross(float3 p0, float3 p1) {
  const float x = p0[1]*p1[2] - p0[2]*p1[1];
  const float y = p0[2]*p1[0] - p0[0]*p1[2];
  const float z = p0[0]*p1[1] - p0[1]*p1[0];
  return {x,y,z};
}
//double4 cross(double4 p0, double4 p1) { assert(0); return {}; }
//double3 cross(double3 p0, double3 p1) { assert(0); return {}; }

// gengeofloat: float, float2, float3, mfloat2, mfloat3, mfloat4, float4,
// gengeodouble: double, double2, double3, double4, mdouble2, mdouble3, mdouble4
//float dot(gengeofloat p0, gengeofloat p1)
//double dot(gengeodouble p0, gengeodouble p1)
float dot(float3 p0, float3 p1) {
  return p0.x()*p1.x() + p0.y()*p1.y() + p0.z()*p1.z();
}

//float length(gengeofloat p)
//double length(gengeodouble p)
float length(float3 p) {
  return sycl::sqrt(p.x()*p.x()+p.y()*p.y()+p.z()*p.z());
}

} // namespace sycl

#endif // _MOTORSYCL_HPP_
