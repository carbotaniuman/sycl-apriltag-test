#include <iostream>
#include <sycl/sycl.hpp>

constexpr auto numWorkItems = 6;
constexpr auto dataSize = 41;

void print(const uint32_t* const data) {
	for (auto i = 0; i < dataSize; i++) {
		std::cout << data[i] << ", ";
	}
	std::cout << "\n";
}

auto selector = sycl::gpu_selector_v;
using op_type = sycl::multiplies<uint32_t>;
auto op = op_type{};
auto identity = sycl::known_identity_v<op_type, uint32_t>;

void fill_data(uint32_t* first, size_t size) {
	for (size_t i = 0; i < size; i++) {
		if (i % 4 == 0) {
			first[i] = 2;
		} else {
			first[i] = 1;
		}
	}
}

void foo() {
	sycl::queue queue{selector};
	std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

	auto* const data = sycl::malloc_shared<uint32_t>(dataSize, queue);
	fill_data(data, dataSize);

	std::cout << "Pre-scan Acpp: "; print(data);

	std::vector<uint32_t> test(dataSize);
    fill_data(test.data(), dataSize);

	std::cout << "Pre-scan Stdl: "; print(test.data());

	std::inclusive_scan(
		test.begin(),
		test.end(),
		test.begin(),
        op,
		identity
	);

	queue.parallel_for(sycl::nd_range<1>{numWorkItems, numWorkItems},
		[=](const sycl::nd_item<1>& item) {
			sycl::joint_inclusive_scan(
				item.get_sub_group(),
				data,
				data + dataSize,
				data,
                op
			);
		}
	);

	queue.wait();

	std::cout << "Post-scan Acpp: "; print(data);
	std::cout << "Post-scan Stdl: "; print(test.data());

    queue.fill(data, uint32_t{1}, dataSize);
	queue.wait();

    queue.parallel_for(sycl::nd_range<1>{numWorkItems, numWorkItems},
		[=](const sycl::nd_item<1>& item) {
            auto ret = sycl::inclusive_scan_over_group(
                item.get_sub_group(),
                data[item.get_local_linear_id()],
                op
            );
            data[item.get_local_linear_id()] = ret;
		}
	);
    queue.wait();
    std::cout << "Over-grup Acpp: "; print(test.data());

	sycl::free(data, queue);
}

void bar() {
	sycl::queue queue{selector};
	std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

	auto* const data = sycl::malloc_shared<uint32_t>(dataSize, queue);
	fill_data(data, dataSize);

	std::cout << "Pre-scan Acpp: "; print(data);

	std::vector<uint32_t> test(dataSize);
    fill_data(test.data(), dataSize);

	std::cout << "Pre-scan Stdl: "; print(test.data());

	std::exclusive_scan(
		test.begin(),
		test.end(),
		test.begin(),
		identity,
        op
	);

	queue.parallel_for(sycl::nd_range<1>{numWorkItems, numWorkItems},
		[=](const sycl::nd_item<1>& item) {
			sycl::joint_exclusive_scan(
				item.get_sub_group(),
				data,
				data + dataSize,
				data,
				op
			);
		}
	);

	queue.wait();

	std::cout << "Post-scan Acpp: "; print(data);
	std::cout << "Post-scan Stdl: "; print(test.data());

    queue.fill(data, uint32_t{1}, dataSize);
	queue.wait();

    queue.parallel_for(sycl::nd_range<1>{numWorkItems, numWorkItems},
		[=](const sycl::nd_item<1>& item) {
            auto ret = sycl::exclusive_scan_over_group(
                item.get_sub_group(),
                data[item.get_local_linear_id()],
                op
            );
            data[item.get_local_linear_id()] = ret;
		}
	);
    queue.wait();
    std::cout << "Over-grup Acpp: "; print(test.data());

	sycl::free(data, queue);
}

int main() {
	for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }
    foo();
    bar();
}
