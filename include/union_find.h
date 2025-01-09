#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <sycl/sycl.hpp>

template <sycl::memory_scope Scope,
          sycl::access::address_space Space =
              sycl::access::address_space::generic_space>
class UnionFind {
    using atomic_elem_ref =
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, Scope, Space>;
    static_assert(atomic_elem_ref::is_always_lock_free);

    uint32_t *m_buffer;

public:
    UnionFind(uint32_t *buffer) : m_buffer(buffer) {}

    uint32_t find(uint32_t n) {
        while (true) {
            atomic_elem_ref ref{m_buffer[n]};

            if (auto next = ref.load(); next != n) {
                n = next;
            } else {
                break;
            }
        }

        return n;
    }

    uint32_t find_compress(uint32_t n) {
        atomic_elem_ref orig{m_buffer[n]};

        while (true) {
            atomic_elem_ref ref{m_buffer[n]};

            if (auto next = ref.load(); next != n) {
                n = next;
                orig.fetch_min(n);
            } else {
                break;
            }
        }

        return n;
    }

    void merge(uint32_t a, uint32_t b) {
        bool done = false;

        while (true) {
            a = find(a);
            b = find(b);

            if (a < b) {
                atomic_elem_ref ref{m_buffer[b]};
                uint32_t old = ref.fetch_min(a);

                if (old == b) {
                    break;
                }
                b = old;
            } else if (b < a) {
                atomic_elem_ref ref{m_buffer[a]};
                uint32_t old = ref.fetch_min(b);

                if (old == a) {
                    break;
                }
                a = old;
            } else {
                break;
            }
        }
    }
};

#endif
