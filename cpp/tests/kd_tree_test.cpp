#include "kd_tree.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double l2(const std::vector<float>& a, const std::vector<float>& b) {
    double d2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        d2 += d * d;
    }
    return std::sqrt(d2);
}

std::vector<std::pair<float, uint64_t>> bruteKNN(const std::vector<KDEntry>& entries, const std::vector<float>& q, std::size_t k) {
    std::vector<std::pair<float, uint64_t>> rows;
    rows.reserve(entries.size());
    for (const auto& e : entries) {
        rows.push_back({static_cast<float>(l2(e.values, q)), e.id});
    }
    const std::size_t limit = std::min(k, rows.size());
    std::partial_sort(rows.begin(), rows.begin() + limit, rows.end(), [](const auto& l, const auto& r) {
        return l.first < r.first;
    });
    rows.resize(limit);
    return rows;
}

}  // namespace

int main() {
    {
        KDTree tree(2);
        std::vector<KDEntry> entries = {
            {0, {0.0f, 0.0f}},
            {1, {10.0f, 0.0f}},
            {2, {20.0f, 0.0f}},
            {3, {30.0f, 0.0f}},
        };
        tree.build(entries);

        auto knn = tree.searchKNN({9.0f, 0.0f}, 2);
        assert(knn.size() == 2);
        assert(knn[0].second == 1);
        assert(knn[1].second == 0 || knn[1].second == 2);
    }

    {
        KDTree tree(3);
        std::vector<KDEntry> entries;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (uint64_t i = 0; i < 200; ++i) {
            entries.push_back({i, {dist(rng), dist(rng), dist(rng)}});
        }
        tree.build(entries);

        const std::vector<float> q = {0.25f, -0.5f, 1.25f};
        const auto kd = tree.searchKNN(q, 10);
        const auto brute = bruteKNN(entries, q, 10);

        assert(kd.size() == brute.size());
        for (std::size_t i = 0; i < kd.size(); ++i) {
            assert(std::fabs(kd[i].first - brute[i].first) < 1e-4f);
        }
    }

    {
        KDTree tree(2);
        std::vector<KDEntry> entries = {
            {100, {1.0f, 1.0f}},
            {200, {1.0f, 1.0f}},
            {300, {2.0f, 2.0f}},
        };
        tree.build(entries);

        auto exact = tree.searchExactPoint({1.0f, 1.0f});
        assert(exact.size() == 2);
        assert(exact[0] == 100);
        assert(exact[1] == 200);

        auto none = tree.searchExactPoint({9.0f, 9.0f});
        assert(none.empty());
    }

    std::printf("KD-tree test passed.\n");
    return 0;
}
