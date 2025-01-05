#include <sycl/sycl.hpp>

#include <iostream>

template <size_t N> class Foo {
    static constexpr std::array<std::array<int, N>, 2> DATA = {
        std::array{0, 100, 200, 300}, std::array{0, 1001, 2002, 3003}};

public:
    static int lookup(size_t n) {
        auto cur = DATA[(n / 4) % 2];
        return cur[n % 4];
    }
};

int main(int argc, char *argv[]) {
    sycl::queue q;
    if (argc == 1) {
        q = sycl::queue{sycl::cpu_selector_v,
                        sycl::property::queue::in_order{}};
    } else {
        q = sycl::queue{sycl::gpu_selector_v,
                        sycl::property::queue::in_order{}};
    }

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    int *data = sycl::malloc_device<int>(1024, q);
    q.parallel_for(sycl::range{1024}, [=](auto idx) {
         const auto &foo = Foo<4>::lookup(idx);
         data[idx] = idx + foo;
     }).wait();

    std::vector<int> result(1024);
    q.memcpy(result.data(), data, result.size()).wait();
    for (int i = 0; i < 8; ++i) {
        std::cout << result[i] << std::endl;
    }
}