#include <hipSYCL/algorithms/algorithm.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

void dumpToCsv(const uint32_t *b, size_t size, const std::string &filename) {
    // Open the CSV file for writing
    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // Loop over the array and write each point to the CSV
    for (size_t i = 0; i < size; ++i) {
        csvFile << b[i] << '\n';
    }

    // Close the CSV file
    csvFile.close();

    std::cout << "CSV file '" << filename << "' has been written.\n";
}

int main(int argc, char *argv[]) {
    sycl::queue q;
    if (argc == 1) {
        q = sycl::queue{sycl::default_selector_v,
                        sycl::property::queue::in_order{}};
    } else {
        q = sycl::queue{sycl::gpu_selector_v,
                        sycl::property::queue::in_order{}};
    }
    acpp::algorithms::util::allocation_cache alloc_cache{
        acpp::algorithms::util::allocation_type::device};

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    size_t elems = 512 * 512;

    auto values_buffer = sycl::malloc_shared<uint32_t>(elems, q);
    for (int i = 0; i < elems; i++) {
        values_buffer[i] = 0;
    }

    auto summed_buffer = sycl::malloc_shared<uint32_t>(elems, q);

    acpp::algorithms::util::allocation_group test_scratch{&alloc_cache,
                                                          q.get_device()};

    acpp::algorithms::inclusive_scan(q, test_scratch, values_buffer,
                                     values_buffer + elems, summed_buffer)
        .wait();
    dumpToCsv(summed_buffer, elems, "foo.csv");

    for (int i = 0; i < elems; i++) {
        // if (i < 128) {
        if (summed_buffer[i] != 0) {
            std::cout << "mismatch a" << std::endl;
            std::cout << summed_buffer[i] << " ! " << i << std::endl;
            __builtin_trap();
        }
        // } else {
        //     if ((i / 128) * 128 != summed_buffer[i]) {
        //         std::cout << "mismatch b" << std::endl;
        //         std::cout << summed_buffer[i] << " ! " << i << std::endl;
        //         __builtin_trap();
        //     }
        // }
    }
}
