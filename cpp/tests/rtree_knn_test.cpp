// rtree_knn_test.cpp
// Tests the branch-and-bound KNN search on a 2-dimensional R-tree.
//
// Setup: insert 10 points along the X-axis at (i*10, 0) with value = i,
//        for i = 0..9. That gives x positions 0, 10, 20, ..., 90.
//
// Query 1 – exact k=3 from (45, 0):
//   nearest: x=40 (dist=5, val=4), x=50 (dist=5, val=5), x=30 (dist=15, val=3)
//   [x=60 also at dist=15 — tie; both outcomes acceptable for 3rd slot]
//
// Query 2 – k=1 from (0, 0):
//   nearest: exactly (0,0), value=0, distance=0.
//
// Query 3 – k=10 from (1000, 0):
//   must return all 10 inserted points, farthest = (0,0) at dist=1000.
//
// Query 4 – high-dimension check: insert 20 points in 128 dims, query
//   nearest-2 and verify both returned distances are the smallest possible.

#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "rtree_node.h"
#include "storage_manager.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

static std::string make_temp_path(const char* tag) {
    return std::string("/tmp/rtree_knn_test_") + tag + ".db";
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}

// Brute-force k nearest from a point set.
static std::vector<std::pair<float, uint64_t>> brute_knn(
        const std::vector<std::vector<float>>& points,
        const std::vector<uint64_t>& values,
        const std::vector<float>& query,
        std::size_t k) {
    std::vector<std::pair<float, uint64_t>> dists;
    dists.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        dists.emplace_back(euclidean(query, points[i]), values[i]);
    }
    std::sort(dists.begin(), dists.end());
    if (dists.size() > k) dists.resize(k);
    return dists;
}

// ---------------------------------------------------------------------------
// test 1 – 2D axis points, structured queries
// ---------------------------------------------------------------------------
static void test_2d_axis(const char* db_path) {
    std::remove(db_path);
    StorageManager sm;
    sm.open(db_path);
    BufferPoolManager bpm(64, &sm);
    RTreeIndex idx(&bpm, static_cast<uint16_t>(2));
    for (int i = 0; i < 10; ++i) {
        idx.insertPoint({static_cast<float>(i * 10), 0.0f}, static_cast<uint64_t>(i));
    }

    // --- Query 1: k=3, query at (45, 0) ---
    auto res = idx.searchKNN({45.0f, 0.0f}, 3);
    assert(res.size() == 3 && "expected 3 results");

    // The two closest are at distance 5 (x=40, x=50)
    assert(res[0].first <= res[1].first && res[1].first <= res[2].first &&
           "results must be sorted nearest-first");
    assert(std::abs(res[0].first - 5.0f) < 1e-3f && "1st result distance should be 5");
    assert(std::abs(res[1].first - 5.0f) < 1e-3f && "2nd result distance should be 5");
    assert(res[2].first >= 14.0f && "3rd result must be farther than 14");

    // values 4 and 5 (x=40, x=50) must be in the top-2 results
    std::set<uint64_t> top2 = {res[0].second, res[1].second};
    assert(top2.count(4) && top2.count(5) && "top-2 must be values 4 and 5");

    // --- Query 2: k=1, query at origin ---
    auto res2 = idx.searchKNN({0.0f, 0.0f}, 1);
    assert(res2.size() == 1);
    assert(res2[0].second == 0 && "nearest to origin must be value 0");
    assert(std::abs(res2[0].first) < 1e-4f && "distance to origin must be ~0");

    // --- Query 3: k=10 from far away ---
    auto res3 = idx.searchKNN({1000.0f, 0.0f}, 10);
    assert(res3.size() == 10 && "must return all 10 points");
    // Farthest point is (0,0) at dist 1000
    assert(std::abs(res3.back().first - 1000.0f) < 1e-2f && "farthest must be ~1000");

    std::remove(db_path);
}

// ---------------------------------------------------------------------------
// test 2 – brute-force comparison on random 2D data
// ---------------------------------------------------------------------------
static void test_vs_brute_force(const char* db_path) {
    std::remove(db_path);
    StorageManager sm;
    sm.open(db_path);
    BufferPoolManager bpm(128, &sm);
    RTreeIndex idx(&bpm, static_cast<uint16_t>(2));

    const int N = 50;
    std::vector<std::vector<float>> points(N, std::vector<float>(2));
    std::vector<uint64_t> values(N);
    // Deterministic pseudo-random placement
    for (int i = 0; i < N; ++i) {
        points[i][0] = static_cast<float>((i * 37 + 13) % 100);
        points[i][1] = static_cast<float>((i * 53 + 7)  % 100);
        values[i]    = static_cast<uint64_t>(i);
        idx.insertPoint(points[i], values[i]);
    }

    // Several query points
    std::vector<std::vector<float>> queries = {
        {50.0f, 50.0f},
        {0.0f,  0.0f},
        {99.0f, 99.0f},
        {25.0f, 75.0f},
    };
    const std::size_t K = 5;

    for (const auto& q : queries) {
        auto expected = brute_knn(points, values, q, K);
        auto actual   = idx.searchKNN(q, K);

        assert(actual.size() == K);
        // Distances must match brute-force (sorted)
        for (std::size_t i = 0; i < K; ++i) {
            assert(std::abs(actual[i].first - expected[i].first) < 1e-2f &&
                   "KNN distance mismatch vs brute-force");
        }
    }

    std::remove(db_path);
}

// ---------------------------------------------------------------------------
// test 3 – 128-dim points
// ---------------------------------------------------------------------------
static void test_128d(const char* db_path) {
    std::remove(db_path);
    StorageManager sm;
    sm.open(db_path);
    BufferPoolManager bpm(256, &sm);
    RTreeIndex idx(&bpm, static_cast<uint16_t>(128));

    const int N = 20;
    std::vector<std::vector<float>> points(N, std::vector<float>(128, 0.0f));
    for (int i = 0; i < N; ++i) {
        points[i][0] = static_cast<float>(i);   // only dimension 0 varies
        idx.insertPoint(points[i], static_cast<uint64_t>(i));
    }

    // Query nearest-2 from (9.4, 0, 0, ...)
    std::vector<float> query(128, 0.0f);
    query[0] = 9.4f;
    auto res = idx.searchKNN(query, 2);

    assert(res.size() == 2);
    // The two nearest along dim-0 should be 9 and 10.
    std::set<uint64_t> vals = {res[0].second, res[1].second};
    assert(vals.count(9) && vals.count(10) && "nearest-2 in 128D must be values 9 and 10");
    assert(std::abs(res[0].first - 0.4f) < 1e-4f && "1st dist should be ~0.4");
    assert(std::abs(res[1].first - 0.6f) < 1e-4f && "2nd dist should be ~0.6");

    std::remove(db_path);
}

int main() {
    test_2d_axis(make_temp_path("2d").c_str());
    test_vs_brute_force(make_temp_path("brute").c_str());
    test_128d(make_temp_path("128d").c_str());

    std::printf("R-tree KNN search test passed.\n");
    return 0;
}
