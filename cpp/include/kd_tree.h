#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

struct KDEntry {
    uint64_t id;
    std::vector<float> values;
};

class KDTree {
public:
    explicit KDTree(uint16_t dimensions);

    void build(const std::vector<KDEntry>& entries);

    // Returns up to k (distance, value) pairs sorted nearest-first.
    std::vector<std::pair<float, uint64_t>> searchKNN(
        const std::vector<float>& query, std::size_t k) const;

    // Returns all IDs whose coordinates exactly match query.
    std::vector<uint64_t> searchExactPoint(const std::vector<float>& query) const;

    std::size_t size() const;
    uint16_t dimensions() const;

private:
    struct Node {
        uint32_t entry_index;
        uint16_t axis;
        int left;
        int right;
    };

    uint16_t dimensions_;
    std::vector<KDEntry> entries_;
    std::vector<Node> nodes_;
    int root_;

    int buildRecursive(std::vector<uint32_t>& indices, int begin, int end, uint16_t depth);
    double squaredDistance(const std::vector<float>& a, const std::vector<float>& b) const;

    void searchKNNRecursive(
        int node_index,
        const std::vector<float>& query,
        std::size_t k,
        std::priority_queue<std::pair<double, uint64_t>>& best) const;

    void searchExactRecursive(int node_index, const std::vector<float>& query, std::vector<uint64_t>& matches) const;

    void validateQueryDimensions(const std::vector<float>& query) const;
};
