#pragma once

#include "buffer_pool_manager.h"
#include "rtree_node.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

class RTreeIndex {
public:
    // Create a brand-new index. Allocates a metadata page as the first page.
    RTreeIndex(BufferPoolManager* buffer_pool_manager, uint16_t dimensions);

    // Reopen an existing index from a previously persisted metadata page.
    RTreeIndex(BufferPoolManager* buffer_pool_manager, uint32_t meta_page_id);

    void insert(const BoundingBox& mbr, uint64_t value);
    void insertPoint(const std::vector<float>& coordinates, uint64_t value);

    // Returns up to k (distance, value) pairs sorted nearest-first.
    std::vector<std::pair<float, uint64_t>> searchKNN(
        const std::vector<float>& query, std::size_t k) const;

    uint32_t getRootPageId() const;
    uint32_t getMetaPageId() const;
    uint16_t getDimensions() const;
    std::size_t getHeight() const;

private:
    struct SplitResult {
        bool did_split;
        uint32_t right_page_id;
        BoundingBox right_mbr;
    };

    BufferPoolManager* buffer_pool_manager_;
    uint16_t dimensions_;
    uint32_t root_page_id_;
    std::size_t height_;
    uint32_t meta_page_id_;
    mutable std::unordered_map<uint32_t, RTreeNodePage> node_cache_;

    const RTreeNodePage& loadNode(uint32_t page_id) const;
    void writeNode(const RTreeNodePage& node) const;
    RTreeNodePage allocateNode(bool is_leaf) const;
    void writeMetadata() const;

    SplitResult insertRecursive(uint32_t page_id, const RTreeEntry& entry);
    SplitResult splitAndPersistNode(const RTreeNodePage& node, const std::vector<RTreeEntry>& entries);

    std::size_t chooseSubtree(const std::vector<RTreeEntry>& entries, const BoundingBox& target) const;
    void updateChildParentLinks(const RTreeNodePage& node) const;
    void validateEntryDimensions(const BoundingBox& mbr) const;
};
