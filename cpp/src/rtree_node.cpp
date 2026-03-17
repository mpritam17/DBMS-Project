#include "rtree_node.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace {

constexpr uint32_t kInvalidLinkPageId = std::numeric_limits<uint32_t>::max();

}

BoundingBox::BoundingBox(std::vector<float> lower, std::vector<float> upper)
    : lower_bounds(std::move(lower)), upper_bounds(std::move(upper)) {
    if (lower_bounds.size() != upper_bounds.size()) {
        throw std::invalid_argument("Bounding box dimension mismatch");
    }
    for (std::size_t index = 0; index < lower_bounds.size(); ++index) {
        if (lower_bounds[index] > upper_bounds[index]) {
            throw std::invalid_argument("Bounding box lower bound exceeds upper bound");
        }
    }
}

std::size_t BoundingBox::dimensions() const {
    return lower_bounds.size();
}

double BoundingBox::hyperVolume() const {
    if (lower_bounds.empty()) {
        return 0.0;
    }

    double volume = 1.0;
    for (std::size_t index = 0; index < lower_bounds.size(); ++index) {
        volume *= static_cast<double>(upper_bounds[index] - lower_bounds[index]);
    }
    return volume;
}

BoundingBox BoundingBox::expandedToInclude(const BoundingBox& other) const {
    if (dimensions() != other.dimensions()) {
        throw std::invalid_argument("Bounding box dimension mismatch");
    }

    std::vector<float> lower = lower_bounds;
    std::vector<float> upper = upper_bounds;

    for (std::size_t index = 0; index < lower.size(); ++index) {
        lower[index] = std::min(lower[index], other.lower_bounds[index]);
        upper[index] = std::max(upper[index], other.upper_bounds[index]);
    }

    return BoundingBox(std::move(lower), std::move(upper));
}

double BoundingBox::enlargementToInclude(const BoundingBox& other) const {
    BoundingBox expanded = expandedToInclude(other);
    return expanded.hyperVolume() - hyperVolume();
}

BoundingBox BoundingBox::point(const std::vector<float>& coordinates) {
    return BoundingBox(coordinates, coordinates);
}

double BoundingBox::logHyperVolume() const {
    if (lower_bounds.empty()) {
        return 0.0;
    }
    double log_vol = 0.0;
    for (std::size_t i = 0; i < lower_bounds.size(); ++i) {
        double edge = static_cast<double>(upper_bounds[i]) - static_cast<double>(lower_bounds[i]);
        if (edge <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        log_vol += std::log(edge);
    }
    return log_vol;
}

// Returns log(V_expanded / V_original).  A smaller value means the MBR needs
// less relative growth to accommodate `other`, which is the correct high-dim
// criterion when the absolute product of edge lengths underflows to 0.
double BoundingBox::logEnlargementRatio(const BoundingBox& other) const {
    if (dimensions() != other.dimensions()) {
        throw std::invalid_argument("Bounding box dimension mismatch");
    }
    double ratio = 0.0;
    for (std::size_t i = 0; i < lower_bounds.size(); ++i) {
        double orig_lo = static_cast<double>(lower_bounds[i]);
        double orig_hi = static_cast<double>(upper_bounds[i]);
        double exp_lo  = std::min(orig_lo, static_cast<double>(other.lower_bounds[i]));
        double exp_hi  = std::max(orig_hi, static_cast<double>(other.upper_bounds[i]));
        double orig_edge = orig_hi - orig_lo;
        double exp_edge  = exp_hi  - exp_lo;
        if (orig_edge <= 0.0 && exp_edge <= 0.0) {
            continue;   // both zero — no change in this dimension
        }
        if (orig_edge <= 0.0) {
            return std::numeric_limits<double>::infinity();  // zero→nonzero always costs more
        }
        ratio += std::log(exp_edge / orig_edge);
    }
    return ratio;
}

RTreeNodePage::RTreeNodePage(uint32_t page_id, uint16_t dimensions, bool is_leaf) {
    if (dimensions == 0) {
        throw std::invalid_argument("R-tree node must have at least one dimension");
    }

    data_.resize(PageLayout::kPageSize, 0);

    PageHeader* header = pageHeaderMut();
    header->magic = RTREE_PAGE_MAGIC;
    header->page_type = is_leaf ? RTREE_LEAF_PAGE_TYPE : RTREE_INTERNAL_PAGE_TYPE;
    header->flags = 0;
    header->page_id = page_id;
    header->item_count = 0;

    NodeHeader* node = nodeHeaderMut();
    node->dimensions = dimensions;
    node->entry_count = 0;
    node->max_entries = static_cast<uint16_t>((PageLayout::kPageSize - PageLayout::kHeaderSize - sizeof(NodeHeader)) / entrySize());
    node->reserved0 = 0;
    node->parent_page_id = kInvalidLinkPageId;
    node->next_leaf_page_id = kInvalidLinkPageId;
    node->reserved1 = 0;
    node->reserved2 = 0;
    node->reserved3 = 0;
    node->reserved4 = 0;

    if (node->max_entries == 0) {
        throw std::invalid_argument("R-tree page cannot fit any entries at this dimension");
    }

    refreshFreeSpaceMetadata();
}

RTreeNodePage::RTreeNodePage(const std::vector<uint8_t>& raw_data) : data_(raw_data) {
    if (data_.size() != PageLayout::kPageSize) {
        throw std::runtime_error("Invalid page size for R-tree node");
    }

    const PageHeader* header = pageHeader();
    if (header->magic != RTREE_PAGE_MAGIC) {
        throw std::runtime_error("Invalid R-tree page magic");
    }
    if (header->page_type != RTREE_LEAF_PAGE_TYPE && header->page_type != RTREE_INTERNAL_PAGE_TYPE) {
        throw std::runtime_error("Invalid R-tree page type");
    }

    const NodeHeader* node = nodeHeader();
    if (node->dimensions == 0) {
        throw std::runtime_error("Invalid R-tree node dimensions");
    }
    if (node->entry_count > node->max_entries) {
        throw std::runtime_error("Corrupt R-tree node entry count");
    }
}

bool RTreeNodePage::isLeaf() const {
    return pageHeader()->page_type == RTREE_LEAF_PAGE_TYPE;
}

uint32_t RTreeNodePage::getPageId() const {
    return pageHeader()->page_id;
}

uint16_t RTreeNodePage::getDimensions() const {
    return nodeHeader()->dimensions;
}

uint16_t RTreeNodePage::getEntryCount() const {
    return nodeHeader()->entry_count;
}

uint16_t RTreeNodePage::getMaxEntries() const {
    return nodeHeader()->max_entries;
}

uint32_t RTreeNodePage::getParentPageId() const {
    return nodeHeader()->parent_page_id;
}

void RTreeNodePage::setParentPageId(uint32_t parent_page_id) {
    nodeHeaderMut()->parent_page_id = parent_page_id;
}

uint32_t RTreeNodePage::getNextLeafPageId() const {
    return nodeHeader()->next_leaf_page_id;
}

void RTreeNodePage::setNextLeafPageId(uint32_t next_leaf_page_id) {
    nodeHeaderMut()->next_leaf_page_id = next_leaf_page_id;
}

bool RTreeNodePage::addEntry(const BoundingBox& mbr, uint64_t value) {
    validateBoundingBox(mbr);

    NodeHeader* node = nodeHeaderMut();
    if (node->entry_count >= node->max_entries) {
        return false;
    }

    const std::size_t offset = entryOffset(node->entry_count);
    const std::size_t coordinate_bytes = static_cast<std::size_t>(node->dimensions) * sizeof(float);

    std::memcpy(data_.data() + offset, mbr.lower_bounds.data(), coordinate_bytes);
    std::memcpy(data_.data() + offset + coordinate_bytes, mbr.upper_bounds.data(), coordinate_bytes);
    std::memcpy(data_.data() + offset + (2 * coordinate_bytes), &value, sizeof(value));

    node->entry_count++;
    pageHeaderMut()->item_count = node->entry_count;
    refreshFreeSpaceMetadata();
    return true;
}

RTreeEntry RTreeNodePage::getEntry(uint16_t index) const {
    const NodeHeader* node = nodeHeader();
    if (index >= node->entry_count) {
        throw std::out_of_range("R-tree entry index out of range");
    }

    const std::size_t dims = node->dimensions;
    const std::size_t offset = entryOffset(index);
    const std::size_t coordinate_bytes = dims * sizeof(float);

    std::vector<float> lower(dims);
    std::vector<float> upper(dims);
    uint64_t value = 0;

    std::memcpy(lower.data(), data_.data() + offset, coordinate_bytes);
    std::memcpy(upper.data(), data_.data() + offset + coordinate_bytes, coordinate_bytes);
    std::memcpy(&value, data_.data() + offset + (2 * coordinate_bytes), sizeof(value));

    return {BoundingBox(std::move(lower), std::move(upper)), value};
}

std::vector<RTreeEntry> RTreeNodePage::getEntries() const {
    std::vector<RTreeEntry> entries;
    entries.reserve(getEntryCount());
    for (uint16_t index = 0; index < getEntryCount(); ++index) {
        entries.push_back(getEntry(index));
    }
    return entries;
}

BoundingBox RTreeNodePage::computeNodeMBR() const {
    if (getEntryCount() == 0) {
        throw std::runtime_error("Cannot compute MBR for empty R-tree node");
    }

    BoundingBox aggregate = getEntry(0).mbr;
    for (uint16_t index = 1; index < getEntryCount(); ++index) {
        aggregate = aggregate.expandedToInclude(getEntry(index).mbr);
    }
    return aggregate;
}

PageHeader* RTreeNodePage::pageHeaderMut() {
    return reinterpret_cast<PageHeader*>(data_.data());
}

const PageHeader* RTreeNodePage::pageHeader() const {
    return reinterpret_cast<const PageHeader*>(data_.data());
}

RTreeNodePage::NodeHeader* RTreeNodePage::nodeHeaderMut() {
    return reinterpret_cast<NodeHeader*>(data_.data() + PageLayout::kHeaderSize);
}

const RTreeNodePage::NodeHeader* RTreeNodePage::nodeHeader() const {
    return reinterpret_cast<const NodeHeader*>(data_.data() + PageLayout::kHeaderSize);
}

std::size_t RTreeNodePage::entrySize() const {
    return static_cast<std::size_t>(2 * nodeHeader()->dimensions) * sizeof(float) + sizeof(uint64_t);
}

std::size_t RTreeNodePage::entryOffset(uint16_t index) const {
    return PageLayout::kHeaderSize + sizeof(NodeHeader) + (static_cast<std::size_t>(index) * entrySize());
}

void RTreeNodePage::validateBoundingBox(const BoundingBox& mbr) const {
    if (mbr.dimensions() != getDimensions()) {
        throw std::invalid_argument("Bounding box dimensions do not match node dimensions");
    }
}

void RTreeNodePage::refreshFreeSpaceMetadata() {
    PageHeader* header = pageHeaderMut();
    const NodeHeader* node = nodeHeader();
    const std::size_t used_bytes = PageLayout::kHeaderSize + sizeof(NodeHeader) + (static_cast<std::size_t>(node->entry_count) * entrySize());
    header->free_space_offset = static_cast<uint16_t>(used_bytes);
    header->free_space_bytes = static_cast<uint16_t>(PageLayout::kPageSize - used_bytes);
    header->item_count = node->entry_count;
}