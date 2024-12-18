#ifndef COMBINADICS_H
#define COMBINADICS_H

#include <array>
#include <algorithm>
#include <cstddef>

// A class to implement the combinatorial number system.
// https://en.wikipedia.org/wiki/Combinatorial_number_system
// The template paramter T is the data type for the lookup table,
// make sure N choose K fits inside this data type!
//
// T should be an unsigned integer.
template <typename T, size_t N, size_t K, size_t Alignment = 0>
class Combinadics {
    constexpr static std::array<std::array<T, N + 1>, K> get_precomputed_binomials() {
        std::array<std::array<T, N + 1>, K + 1> lookup{};

        for (int n = 0; n <= N; n++) {
            lookup[0][n] = 1;
        }

        for (int k = 1; k <= K; k++) {
            for (int n = 1; n <= N; n++) {
                lookup[k][n] = lookup[k - 1][n - 1] + lookup[k][n - 1];
            }
        }

        // Drop the 0 zero for K beacause it is never used.
        std::array<std::array<T, N + 1>, K> ret{};
        for (int k = 1; k <= K; k++) {
            for (int n = 0; n <= N; n++) {
                ret[k - 1][n] = lookup[k][n];
            }
        }
    
        return ret;
    }

    // Try to convince the compiler to align this
    alignas(Alignment) constexpr static auto PRECOMPUTED_BINOMIALS = get_precomputed_binomials();
public:

    // Any values corresponding to a N' > N will result
    // in an arbitrary invalid (ie non-combination)
    // value being returned! Don't do that.
    std::array<size_t, K> decode(T num) {
        std::array<size_t, K> ret{};

        for (int i = 0; i < K; i++) {
            const auto &cur = PRECOMPUTED_BINOMIALS[K - 1 - i];
            auto found = std::upper_bound(cur.cbegin(), cur.cend(), num);
            if (found == cur.cbegin()) {
                return ret;
            }
            found--;

            num -= *found;
            ret[i] = std::distance(cur.cbegin(), found);
        }

        return ret;
    }
};

#endif
