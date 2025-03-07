#ifndef LABEL_COMPACTER_H
#define LABEL_COMPACTER_H

#include <sycl/sycl.hpp>

class LabelCompacter {
public:
    constexpr static uint64_t EMPTY_KEY_VAL = 0;

    // A zeroed buffer represents an empty map.
    // The 0th entry will never be be initialized.
    // Elems must be a power of 2
    LabelCompacter(uint64_t *buffer)
        : m_buffer(buffer), m_elems(1 << 20) {}


    // Adds the value given to the slot with the given key.
    template <sycl::memory_scope Scope,
              sycl::access::address_space Space =
                  sycl::access::address_space::generic_space>
    size_t lookup(uint64_t key) {
        using atomic_elem_ref =
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, Scope,
                             Space>;
        static_assert(atomic_elem_ref::is_always_lock_free);

        if (key == 0) {
            return 0;
        }

        size_t index = murmur3_finalize(key) & (m_elems - 1);
        size_t first_index = index;

        while (true) {
            // Leave the 0th entry empty.
            if (index != 0) {
                atomic_elem_ref key_ref{m_buffer[index]};

                uint64_t prev_key = key_ref.load();

                if (prev_key == key) {
                    return index;
                } else if (prev_key == EMPTY_KEY_VAL) {
                    if (key_ref.compare_exchange_strong(prev_key, key)) {
                        return index;
                    }
                }
            }

            index = (index + 1) & (m_elems - 1);
            if (index == first_index) {
                break;
            }
        }

        return -1;
    }

private:
    // Just use the Murmur3 finalizer as a hasher
    constexpr uint64_t murmur3_finalize(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;
        return x;
    }

    uint64_t *m_buffer;
    size_t m_elems;
};

#endif
