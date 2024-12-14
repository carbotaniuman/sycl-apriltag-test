#ifndef OPEN_CHAINING_H
#define OPEN_CHAINING_H

#include <sycl/sycl.hpp>

// Just use the Murmur3 finalizer as a hasher
inline uint32_t murmur3_finalize(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}

class HashTable {
public:
    struct Entry {
        uint32_t key;
        uint32_t value;
    };

    constexpr static uint32_t EMPTY_KEY_VAL = 0;

    // A zeroed buffer represents an empty map.
    // The 0th entry will never be be initialized.
    // Elems must be a power of 2
    HashTable(Entry *buffer, size_t elems = 1 << 16)
        : m_buffer(buffer), m_elems(elems) {}

    // Non-atomic, not suited for concurrent use with `insert`
    uint32_t lookup(uint32_t key) {
        uint32_t index = murmur3_finalize(key) & (m_elems - 1);
        uint32_t first_index = index;

        while (index != first_index - 1) {
            uint32_t found_key = m_buffer[index].key;
            if (found_key == key) {
                return m_buffer[index].value;
            } else if (found_key == EMPTY_KEY_VAL) {
                return EMPTY_KEY_VAL;
            }
            index = (index + 1) & (m_elems - 1);
        }

        return EMPTY_KEY_VAL;
    }

    // Adds the value given to the slot with the given key.
    template <sycl::memory_scope Scope,
              sycl::access::address_space Space =
                  sycl::access::address_space::generic_space>
    size_t insert_add(uint32_t key, uint32_t value) {
        using atomic_elem_ref =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, Scope,
                             Space>;
        static_assert(atomic_elem_ref::is_always_lock_free);

        uint32_t index = murmur3_finalize(key) & (m_elems - 1);
        uint32_t first_index = index;

        // If we run out of space just return without writing.
        while (index != first_index - 1) {
            // Leave the 0th entry empty.
            if (index != 0) {
                uint32_t prev_key = EMPTY_KEY_VAL;

                atomic_elem_ref key_ref{m_buffer[index].key};
                atomic_elem_ref value_ref{m_buffer[index].value};

                if (key_ref.compare_exchange_strong(prev_key, key)) {
                    value_ref.fetch_add(value);
                    return index;
                } else {
                    if (prev_key == key) {
                        value_ref.fetch_add(value);
                        return index;
                    }
                }
            }

            index = (index + 1) & (m_elems - 1);
            if (index == first_index) {
                break;
            }
        }

        return EMPTY_KEY_VAL;
    }

private:
    Entry *m_buffer;
    size_t m_elems;
};

#endif
