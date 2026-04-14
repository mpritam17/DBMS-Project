#pragma once

#include "storage_manager.h"
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t RTREE_PAGE_MAGIC = 0x52545245; // "RTRE"
constexpr uint16_t RTREE_LEAF_PAGE_TYPE = 1;
constexpr uint16_t RTREE_INTERNAL_PAGE_TYPE = 2;

struct BoundingBox {
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;

    BoundingBox() = default;
    BoundingBox(std::vector<float> lower, std::vector<float> upper);

    std::size_t dimensions() const;
    double hyperVolume() const;
    BoundingBox expandedToInclude(const BoundingBox& other) const;
    double enlargementToInclude(const BoundingBox& other) const;

    // Log-space volume and enlargement — numerically stable at high dimensions
    // where the product of edge lengths underflows to 0.0.
    double logHyperVolume() const;
    double logEnlargementRatio(const BoundingBox& other) const;

    static BoundingBox point(const std::vector<float>& coordinates);
};

struct RTreeEntry {
    BoundingBox mbr;
    uint64_t value;
};

class RTreeNodePage {
public:
    RTreeNodePage(uint32_t page_id, uint16_t dimensions, bool is_leaf);
    explicit RTreeNodePage(const std::vector<uint8_t>& raw_data);

    const std::vector<uint8_t>& getRawData() const { return data_; }

    bool isLeaf() const;
    uint32_t getPageId() const;
    uint16_t getDimensions() const;
    uint16_t getEntryCount() const;
    uint16_t getMaxEntries() const;
    uint32_t getParentPageId() const;
    void setParentPageId(uint32_t parent_page_id);
    uint32_t getNextLeafPageId() const;
    void setNextLeafPageId(uint32_t next_leaf_page_id);

    bool addEntry(const BoundingBox& mbr, uint64_t value);
    RTreeEntry getEntry(uint16_t index) const;
    std::vector<RTreeEntry> getEntries() const;
    void getEntryView(uint16_t index, const float*& lower, const float*& upper, uint64_t& value) const;
    BoundingBox computeNodeMBR() const;

private:
    struct NodeHeader {
        uint16_t dimensions;
        uint16_t entry_count;
        uint16_t max_entries;
        uint16_t reserved0;
        uint32_t parent_page_id;
        uint32_t next_leaf_page_id;
        uint32_t reserved1;
        uint32_t reserved2;
        uint32_t reserved3;
        uint32_t reserved4;
    };

    static_assert(sizeof(NodeHeader) == 32, "NodeHeader must be 32 bytes");

    std::vector<uint8_t> data_;

    PageHeader* pageHeaderMut();
    const PageHeader* pageHeader() const;
    NodeHeader* nodeHeaderMut();
    const NodeHeader* nodeHeader() const;

    std::size_t entrySize() const;
    std::size_t entryOffset(uint16_t index) const;
    void validateBoundingBox(const BoundingBox& mbr) const;
    void refreshFreeSpaceMetadata();
};