#include "rtree_index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace {

constexpr uint32_t kInvalidPageId = std::numeric_limits<uint32_t>::max();

double boxCenter(const BoundingBox& box, std::size_t axis) {
    return (static_cast<double>(box.lower_bounds[axis]) + static_cast<double>(box.upper_bounds[axis])) / 2.0;
}

// Minimum Euclidean distance from a query point to the surface of an MBR.
// Returns 0 when the query is inside the box.
double minDistToBox(const std::vector<float>& query, const BoundingBox& box) {
    double dist_sq = 0.0;
    for (std::size_t i = 0; i < query.size(); ++i) {
        double q = static_cast<double>(query[i]);
        double lo = static_cast<double>(box.lower_bounds[i]);
        double hi = static_cast<double>(box.upper_bounds[i]);
        double d = 0.0;
        if (q < lo) {
            d = lo - q;
        } else if (q > hi) {
            d = q - hi;
        }
        dist_sq += d * d;
    }
    return std::sqrt(dist_sq);
}

BoundingBox computeEntriesMBR(const std::vector<RTreeEntry>& entries) {
    if (entries.empty()) {
        throw std::runtime_error("Cannot compute MBR of empty entry set");
    }

    BoundingBox combined = entries.front().mbr;
    for (std::size_t index = 1; index < entries.size(); ++index) {
        combined = combined.expandedToInclude(entries[index].mbr);
    }
    return combined;
}

} // namespace

RTreeIndex::RTreeIndex(BufferPoolManager* buffer_pool_manager, uint16_t dimensions)
    : buffer_pool_manager_(buffer_pool_manager), dimensions_(dimensions), root_page_id_(kInvalidPageId), height_(1) {
    if (buffer_pool_manager_ == nullptr) {
        throw std::invalid_argument("RTreeIndex requires a valid buffer pool manager");
    }
    if (dimensions_ == 0) {
        throw std::invalid_argument("RTreeIndex requires at least one dimension");
    }

    RTreeNodePage root = allocateNode(true);
    root_page_id_ = root.getPageId();
}

void RTreeIndex::insert(const BoundingBox& mbr, uint64_t value) {
    validateEntryDimensions(mbr);

    SplitResult split = insertRecursive(root_page_id_, {mbr, value});
    if (!split.did_split) {
        return;
    }

    RTreeNodePage old_root = loadNode(root_page_id_);
    RTreeNodePage new_root = allocateNode(false);
    new_root.addEntry(old_root.computeNodeMBR(), old_root.getPageId());
    new_root.addEntry(split.right_mbr, split.right_page_id);
    writeNode(new_root);

    old_root.setParentPageId(new_root.getPageId());
    writeNode(old_root);

    RTreeNodePage right_child = loadNode(split.right_page_id);
    right_child.setParentPageId(new_root.getPageId());
    writeNode(right_child);

    root_page_id_ = new_root.getPageId();
    height_++;
}

void RTreeIndex::insertPoint(const std::vector<float>& coordinates, uint64_t value) {
    insert(BoundingBox::point(coordinates), value);
}

uint32_t RTreeIndex::getRootPageId() const {
    return root_page_id_;
}

uint16_t RTreeIndex::getDimensions() const {
    return dimensions_;
}

std::size_t RTreeIndex::getHeight() const {
    return height_;
}

std::vector<std::pair<float, uint64_t>> RTreeIndex::searchKNN(
        const std::vector<float>& query, std::size_t k) const {
    if (k == 0) {
        return {};
    }
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query vector dimensions do not match index dimensions");
    }

    // Min-heap: pop the page/entry with the smallest lower-bound distance first.
    struct QueueItem {
        double min_dist;
        uint32_t page_id;
        bool operator>(const QueueItem& other) const { return min_dist > other.min_dist; }
    };
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;
    pq.push({0.0, root_page_id_});

    // Max-heap of size k to track the current k nearest: top = current worst distance.
    using ResultEntry = std::pair<double, uint64_t>;
    std::priority_queue<ResultEntry> results;

    while (!pq.empty()) {
        auto [min_dist, page_id] = pq.top();
        pq.pop();

        // Prune: this subtree cannot improve on our worst accepted distance.
        if (results.size() >= k && min_dist >= results.top().first) {
            continue;
        }

        RTreeNodePage node = loadNode(page_id);

        if (node.isLeaf()) {
            for (const RTreeEntry& entry : node.getEntries()) {
                double dist = minDistToBox(query, entry.mbr);
                if (results.size() < k || dist < results.top().first) {
                    results.push({dist, entry.value});
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        } else {
            for (const RTreeEntry& entry : node.getEntries()) {
                double child_min_dist = minDistToBox(query, entry.mbr);
                if (results.size() < k || child_min_dist < results.top().first) {
                    pq.push({child_min_dist, static_cast<uint32_t>(entry.value)});
                }
            }
        }
    }

    // Drain the max-heap into a vector, then reverse so nearest comes first.
    std::vector<std::pair<float, uint64_t>> output;
    output.reserve(results.size());
    while (!results.empty()) {
        output.emplace_back(static_cast<float>(results.top().first), results.top().second);
        results.pop();
    }
    std::reverse(output.begin(), output.end());
    return output;
}

RTreeNodePage RTreeIndex::loadNode(uint32_t page_id) const {
    Page* page = buffer_pool_manager_->fetchPage(page_id);
    if (page == nullptr) {
        throw std::runtime_error("Failed to fetch R-tree node page");
    }

    page->RLock();
    std::vector<uint8_t> raw(page->getData(), page->getData() + PageLayout::kPageSize);
    page->RUnlock();
    buffer_pool_manager_->unpinPage(page_id, false);
    return RTreeNodePage(raw);
}

void RTreeIndex::writeNode(const RTreeNodePage& node) const {
    Page* page = buffer_pool_manager_->fetchPage(node.getPageId());
    if (page == nullptr) {
        throw std::runtime_error("Failed to fetch R-tree node page for write");
    }

    page->WLock();
    std::memcpy(page->getData(), node.getRawData().data(), PageLayout::kPageSize);
    page->WUnlock();
    buffer_pool_manager_->unpinPage(node.getPageId(), true);
}

RTreeNodePage RTreeIndex::allocateNode(bool is_leaf) const {
    uint32_t page_id = kInvalidPageId;
    Page* page = buffer_pool_manager_->newPage(&page_id);
    if (page == nullptr) {
        throw std::runtime_error("Failed to allocate new R-tree node page");
    }

    RTreeNodePage node(page_id, dimensions_, is_leaf);
    page->WLock();
    std::memcpy(page->getData(), node.getRawData().data(), PageLayout::kPageSize);
    page->WUnlock();
    buffer_pool_manager_->unpinPage(page_id, true);
    return node;
}

RTreeIndex::SplitResult RTreeIndex::insertRecursive(uint32_t page_id, const RTreeEntry& entry) {
    RTreeNodePage node = loadNode(page_id);
    std::vector<RTreeEntry> entries = node.getEntries();

    if (node.isLeaf()) {
        entries.push_back(entry);
        if (entries.size() <= node.getMaxEntries()) {
            RTreeNodePage rewritten(node.getPageId(), dimensions_, true);
            rewritten.setParentPageId(node.getParentPageId());
            rewritten.setNextLeafPageId(node.getNextLeafPageId());
            for (const RTreeEntry& leaf_entry : entries) {
                rewritten.addEntry(leaf_entry.mbr, leaf_entry.value);
            }
            writeNode(rewritten);
            return {false, kInvalidPageId, BoundingBox()};
        }
        return splitAndPersistNode(node, entries);
    }

    std::size_t child_index = chooseSubtree(entries, entry.mbr);
    uint32_t child_page_id = static_cast<uint32_t>(entries[child_index].value);
    SplitResult child_split = insertRecursive(child_page_id, entry);

    RTreeNodePage updated_child = loadNode(child_page_id);
    entries[child_index].mbr = updated_child.computeNodeMBR();
    if (child_split.did_split) {
        entries.push_back({child_split.right_mbr, child_split.right_page_id});
    }

    if (entries.size() <= node.getMaxEntries()) {
        RTreeNodePage rewritten(node.getPageId(), dimensions_, false);
        rewritten.setParentPageId(node.getParentPageId());
        for (const RTreeEntry& internal_entry : entries) {
            rewritten.addEntry(internal_entry.mbr, internal_entry.value);
        }
        writeNode(rewritten);
        updateChildParentLinks(rewritten);
        return {false, kInvalidPageId, BoundingBox()};
    }

    return splitAndPersistNode(node, entries);
}

RTreeIndex::SplitResult RTreeIndex::splitAndPersistNode(const RTreeNodePage& node, const std::vector<RTreeEntry>& entries) {
    if (entries.size() <= node.getMaxEntries()) {
        throw std::invalid_argument("splitAndPersistNode called without overflow");
    }

    std::vector<RTreeEntry> ordered = entries;
    std::size_t axis = chooseSplitAxis(ordered);
    std::sort(ordered.begin(), ordered.end(), [axis](const RTreeEntry& left, const RTreeEntry& right) {
        return boxCenter(left.mbr, axis) < boxCenter(right.mbr, axis);
    });

    const std::size_t split_index = ordered.size() / 2;
    std::vector<RTreeEntry> left_entries(ordered.begin(), ordered.begin() + static_cast<std::ptrdiff_t>(split_index));
    std::vector<RTreeEntry> right_entries(ordered.begin() + static_cast<std::ptrdiff_t>(split_index), ordered.end());

    if (left_entries.empty() || right_entries.empty()) {
        throw std::runtime_error("R-tree split produced an empty partition");
    }

    RTreeNodePage left_node(node.getPageId(), dimensions_, node.isLeaf());
    left_node.setParentPageId(node.getParentPageId());
    if (node.isLeaf()) {
        left_node.setNextLeafPageId(node.getNextLeafPageId());
    }
    for (const RTreeEntry& left_entry : left_entries) {
        left_node.addEntry(left_entry.mbr, left_entry.value);
    }

    RTreeNodePage right_node = allocateNode(node.isLeaf());
    right_node.setParentPageId(node.getParentPageId());
    if (node.isLeaf()) {
        right_node.setNextLeafPageId(node.getNextLeafPageId());
        left_node.setNextLeafPageId(right_node.getPageId());
    }
    for (const RTreeEntry& right_entry : right_entries) {
        right_node.addEntry(right_entry.mbr, right_entry.value);
    }

    writeNode(left_node);
    writeNode(right_node);

    if (!node.isLeaf()) {
        updateChildParentLinks(left_node);
        updateChildParentLinks(right_node);
    }

    return {true, right_node.getPageId(), computeEntriesMBR(right_entries)};
}

std::size_t RTreeIndex::chooseSubtree(const std::vector<RTreeEntry>& entries, const BoundingBox& target) const {
    if (entries.empty()) {
        throw std::runtime_error("Cannot choose subtree from empty internal node");
    }

    std::size_t best_index = 0;
    double best_enlargement = entries[0].mbr.enlargementToInclude(target);
    double best_volume = entries[0].mbr.hyperVolume();

    for (std::size_t index = 1; index < entries.size(); ++index) {
        const double enlargement = entries[index].mbr.enlargementToInclude(target);
        const double volume = entries[index].mbr.hyperVolume();
        if (enlargement < best_enlargement ||
            (std::abs(enlargement - best_enlargement) < 1e-9 && volume < best_volume)) {
            best_index = index;
            best_enlargement = enlargement;
            best_volume = volume;
        }
    }

    return best_index;
}

std::size_t RTreeIndex::chooseSplitAxis(const std::vector<RTreeEntry>& entries) const {
    if (entries.empty()) {
        throw std::runtime_error("Cannot choose split axis for empty entries");
    }

    std::size_t best_axis = 0;
    double best_span = -1.0;

    for (std::size_t axis = 0; axis < dimensions_; ++axis) {
        double min_center = boxCenter(entries[0].mbr, axis);
        double max_center = min_center;
        for (std::size_t index = 1; index < entries.size(); ++index) {
            const double center = boxCenter(entries[index].mbr, axis);
            min_center = std::min(min_center, center);
            max_center = std::max(max_center, center);
        }
        const double span = max_center - min_center;
        if (span > best_span) {
            best_span = span;
            best_axis = axis;
        }
    }

    return best_axis;
}

void RTreeIndex::updateChildParentLinks(const RTreeNodePage& node) const {
    if (node.isLeaf()) {
        return;
    }

    for (const RTreeEntry& entry : node.getEntries()) {
        RTreeNodePage child = loadNode(static_cast<uint32_t>(entry.value));
        child.setParentPageId(node.getPageId());
        writeNode(child);
    }
}

void RTreeIndex::validateEntryDimensions(const BoundingBox& mbr) const {
    if (mbr.dimensions() != dimensions_) {
        throw std::invalid_argument("Inserted bounding box dimensions do not match index dimensions");
    }
}
