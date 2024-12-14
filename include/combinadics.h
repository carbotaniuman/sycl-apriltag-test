#ifndef COMBINADICS_H
#define COMBINADICS_H

#include <algorithm>
#include <array>
#include <cstddef>

// A class to implement the combinatorial number system.
// https://en.wikipedia.org/wiki/Combinatorial_number_system
// The template paramter T is the data type for the lookup table,
// make sure N choose K fits inside this data type!
//
// T should be an unsigned integer.
template <typename T, size_t N, size_t K, size_t Alignment = 0>
class Combinadics {
    constexpr static std::array<std::array<T, N + 1>, K + 1>
    get_precomputed_binomials() {
        std::array<std::array<T, N + 1>, K + 1> lookup{};

        for (int n = 0; n <= N; n++) {
            lookup[0][n] = 1;
        }

        for (int k = 1; k <= K; k++) {
            for (int n = 1; n <= N; n++) {
                lookup[k][n] = lookup[k - 1][n - 1] + lookup[k][n - 1];
            }
        }

        return lookup;
    }

    constexpr static auto PRECOMPUTED_BINOMIALS = get_precomputed_binomials();

public:
    static T n_choose_k(size_t n, size_t k) {
        return PRECOMPUTED_BINOMIALS[k][n];
    }

    // Any values corresponding to a N' > N will result
    // in an arbitrary invalid (ie non-combination)
    // value being returned! Don't do that.
    static std::array<size_t, K> decode(T num) {
        std::array<size_t, K> ret{};

        for (int i = 0; i < K; i++) {
            const auto &cur = PRECOMPUTED_BINOMIALS[K - i];
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
