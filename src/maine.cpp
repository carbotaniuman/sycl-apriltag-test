#include <iostream>
#include <sycl/sycl.hpp>

constexpr auto numWorkItems = 5;
constexpr auto dataSize = 10;

void print(const uint32_t* const data) {
	for (auto i = 0; i < dataSize; i++) {
		std::cout << data[i] << ", ";
	}
	std::cout << "\n";
}

int main() {
	sycl::queue queue{sycl::cpu_selector_v};

	auto* const data = sycl::malloc_host<uint32_t>(dataSize, queue);
	queue.fill(data, uint32_t{1}, dataSize);

	queue.wait();

	std::cout << "Pre-scan Acpp: "; print(data);

	std::vector<uint32_t> test(dataSize);
	for (int i = 0; i < dataSize; i++) {
		test[i] = 1;
	}
	// std::ranges::fill(test, 1);

	std::cout << "Pre-scan Stdlib: "; print(test.data());

	std::exclusive_scan(
		test.begin(),
		test.end(),
		test.begin(),
		0
	);

	queue.parallel_for(sycl::nd_range<1>{numWorkItems, numWorkItems},
		[=](const sycl::nd_item<1>& item) {
			sycl::joint_exclusive_scan(
				item.get_group(),
				data,
				data + dataSize,
				data,
				0,
				sycl::plus<uint32_t>{}
			);
		}
	);

	queue.wait();

	std::cout << "Post-scan Acpp: "; print(data);
	std::cout << "Post-scan Stdlib: "; print(test.data());

	sycl::free(data, queue);

	return 0;
}
