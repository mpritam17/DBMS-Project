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
constexpr uint32_t kMetaMagic = 0x52544958;  // 'RTIX'

struct MetaHeader {
    uint32_t magic;
    uint32_t root_page_id;
    uint32_t height;
    uint16_t dimensions;
    uint8_t  pad[50];
};
static_assert(sizeof(MetaHeader) == 64, "MetaHeader must be 64 bytes");

double boxCenter(const BoundingBox& box, std::size_t axis) {
    return (static_cast<double>(box.lower_bounds[axis]) + static_cast<double>(box.upper_bounds[axis])) / 2.0;
}

double boxMargin(const BoundingBox& box) {
    double margin = 0.0;
    for (std::size_t i = 0; i < box.lower_bounds.size(); ++i) {
        const double edge = static_cast<double>(box.upper_bounds[i]) - static_cast<double>(box.lower_bounds[i]);
        margin += std::max(0.0, edge);
    }
    return margin;
}

double enlargementScore(const BoundingBox& current, const BoundingBox& incoming) {
    const BoundingBox expanded = current.expandedToInclude(incoming);
    return boxMargin(expanded) - boxMargin(current);
}

// Minimum squared Euclidean distance from a query point to the surface of an
// MBR. Returns 0 when the query is inside the box.
double minDistSqToBox(const std::vector<float>& query, const BoundingBox& box) {
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
    return dist_sq;
}

double minDistSqToBox(const std::vector<float>& query, const float* lower, const float* upper, std::size_t dims) {
    double dist_sq = 0.0;
    for (std::size_t i = 0; i < dims; ++i) {
        const double q = static_cast<double>(query[i]);
        const double lo = static_cast<double>(lower[i]);
        const double hi = static_cast<double>(upper[i]);
        double d = 0.0;
        if (q < lo) {
            d = lo - q;
        } else if (q > hi) {
            d = q - hi;
        }
        dist_sq += d * d;
    }
    return dist_sq;
}

bool pointInsideBox(const std::vector<float>& point, const float* lower, const float* upper, std::size_t dims) {
    constexpr double kPointEpsilon = 1e-6;
    for (std::size_t i = 0; i < dims; ++i) {
        const double q = static_cast<double>(point[i]);
        const double lo = static_cast<double>(lower[i]);
        const double hi = static_cast<double>(upper[i]);
        if (q < lo - kPointEpsilon || q > hi + kPointEpsilon) {
            return false;
        }
    }
    return true;
}

bool pointEqualsEntryPoint(const std::vector<float>& point, const float* lower, const float* upper, std::size_t dims) {
    constexpr double kPointEpsilon = 1e-6;
    for (std::size_t i = 0; i < dims; ++i) {
        const double q = static_cast<double>(point[i]);
        const double lo = static_cast<double>(lower[i]);
        const double hi = static_cast<double>(upper[i]);
        if (std::abs(hi - lo) > kPointEpsilon) {
            return false;
        }
        if (std::abs(q - lo) > kPointEpsilon) {
            return false;
        }
    }
    return true;
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
    : buffer_pool_manager_(buffer_pool_manager), dimensions_(dimensions),
      root_page_id_(kInvalidPageId), height_(1), meta_page_id_(kInvalidPageId) {
    if (buffer_pool_manager_ == nullptr) {
        throw std::invalid_argument("RTreeIndex requires a valid buffer pool manager");
    }
    if (dimensions_ == 0) {
        throw std::invalid_argument("RTreeIndex requires at least one dimension");
    }

    // Allocate the metadata page first so callers can reopen the index by this id.
    uint32_t meta_id = kInvalidPageId;
    Page* meta_raw = buffer_pool_manager_->newPage(&meta_id);
    if (meta_raw == nullptr) {
        throw std::runtime_error("Failed to allocate metadata page");
    }
    meta_page_id_ = meta_id;
    buffer_pool_manager_->unpinPage(meta_id, false);

    RTreeNodePage root = allocateNode(true);
    root_page_id_ = root.getPageId();
    writeMetadata();
}

RTreeIndex::RTreeIndex(BufferPoolManager* buffer_pool_manager, uint32_t meta_page_id)
    : buffer_pool_manager_(buffer_pool_manager), dimensions_(0),
      root_page_id_(kInvalidPageId), height_(0), meta_page_id_(meta_page_id) {
    if (buffer_pool_manager_ == nullptr) {
        throw std::invalid_argument("RTreeIndex requires a valid buffer pool manager");
    }

    Page* page = buffer_pool_manager_->fetchPage(meta_page_id);
    if (page == nullptr) {
        throw std::runtime_error("Failed to fetch metadata page");
    }

    page->RLock();
    const uint8_t* data = page->getData();
    const PageHeader* ph = reinterpret_cast<const PageHeader*>(data);
    if (ph->magic != RTREE_PAGE_MAGIC) {
        page->RUnlock();
        buffer_pool_manager_->unpinPage(meta_page_id, false);
        throw std::runtime_error("Invalid metadata page: bad page magic");
    }
    const MetaHeader* mh = reinterpret_cast<const MetaHeader*>(data + sizeof(PageHeader));
    if (mh->magic != kMetaMagic) {
        page->RUnlock();
        buffer_pool_manager_->unpinPage(meta_page_id, false);
        throw std::runtime_error("Invalid metadata page: bad meta magic");
    }
    root_page_id_ = mh->root_page_id;
    height_       = static_cast<std::size_t>(mh->height);
    dimensions_   = mh->dimensions;
    page->RUnlock();
    buffer_pool_manager_->unpinPage(meta_page_id, false);
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
    writeMetadata();
}

void RTreeIndex::insertPoint(const std::vector<float>& coordinates, uint64_t value) {
    insert(BoundingBox::point(coordinates), value);
}

uint32_t RTreeIndex::getRootPageId() const {
    return root_page_id_;
}

uint32_t RTreeIndex::getMetaPageId() const {
    return meta_page_id_;
}

uint16_t RTreeIndex::getDimensions() const {
    return dimensions_;
}

std::size_t RTreeIndex::getHeight() const {
    return height_;
}

void RTreeIndex::clearNodeCache() const {
    node_cache_.clear();
}

void RTreeIndex::writeMetadata() const {
    Page* page = buffer_pool_manager_->fetchPage(meta_page_id_);
    if (page == nullptr) {
        throw std::runtime_error("Failed to fetch metadata page for write");
    }

    page->WLock();
    uint8_t* data = page->getData();
    std::memset(data, 0, PageLayout::kPageSize);

    PageHeader ph{};
    ph.magic     = RTREE_PAGE_MAGIC;
    ph.page_type = 0;
    ph.page_id   = meta_page_id_;
    std::memcpy(data, &ph, sizeof(PageHeader));

    MetaHeader mh{};
    mh.magic        = kMetaMagic;
    mh.root_page_id = root_page_id_;
    mh.height       = static_cast<uint32_t>(height_);
    mh.dimensions   = dimensions_;
    std::memcpy(data + sizeof(PageHeader), &mh, sizeof(MetaHeader));

    page->WUnlock();
    buffer_pool_manager_->unpinPage(meta_page_id_, true);
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
        double min_dist_sq;
        uint32_t page_id;
        bool operator>(const QueueItem& other) const { return min_dist_sq > other.min_dist_sq; }
    };
    std::vector<QueueItem> queue_storage;
    queue_storage.reserve(256);
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq(
        std::greater<QueueItem>(), std::move(queue_storage));
    pq.push({0.0, root_page_id_});

    // Max-heap of size k to track the current k nearest: top = current worst distance.
    using ResultEntry = std::pair<double, uint64_t>;
    std::priority_queue<ResultEntry> results;
    const std::size_t dims = query.size();

    while (!pq.empty()) {
        auto [min_dist_sq, page_id] = pq.top();
        pq.pop();

        // Global stop: pq is a min-heap, so if the best remaining lower bound
        // cannot beat our current worst top-k result, all remaining items are
        // guaranteed non-improving.
        if (results.size() >= k && min_dist_sq >= results.top().first) {
            break;
        }

        const RTreeNodePage& node = loadNode(page_id);
        const uint16_t entry_count = node.getEntryCount();

        if (node.isLeaf()) {
            for (uint16_t i = 0; i < entry_count; ++i) {
                const float* lower = nullptr;
                const float* upper = nullptr;
                uint64_t value = 0;
                node.getEntryView(i, lower, upper, value);
                const double dist_sq = minDistSqToBox(query, lower, upper, dims);
                if (results.size() < k || dist_sq < results.top().first) {
                    results.push({dist_sq, value});
                    if (results.size() > k) {
                        results.pop();
                    }
                }
            }
        } else {
            for (uint16_t i = 0; i < entry_count; ++i) {
                const float* lower = nullptr;
                const float* upper = nullptr;
                uint64_t value = 0;
                node.getEntryView(i, lower, upper, value);
                const double child_min_dist_sq = minDistSqToBox(query, lower, upper, dims);
                if (results.size() < k || child_min_dist_sq < results.top().first) {
                    pq.push({child_min_dist_sq, static_cast<uint32_t>(value)});
                }
            }
        }
    }

    // Drain the max-heap into a vector, then reverse so nearest comes first.
    std::vector<std::pair<float, uint64_t>> output;
    output.reserve(results.size());
    while (!results.empty()) {
        output.emplace_back(static_cast<float>(std::sqrt(results.top().first)), results.top().second);
        results.pop();
    }
    std::reverse(output.begin(), output.end());
    return output;
}

std::vector<uint64_t> RTreeIndex::searchPoint(
        const std::vector<float>& point,
        PointSearchMetrics* metrics) const {
    if (point.size() != dimensions_) {
        throw std::invalid_argument("Point dimensions do not match index dimensions");
    }

    PointSearchMetrics local_metrics{};
    std::vector<uint64_t> matches;
    std::vector<uint32_t> stack;
    stack.push_back(root_page_id_);

    while (!stack.empty()) {
        const uint32_t page_id = stack.back();
        stack.pop_back();

        ++local_metrics.nodes_visited;
        const RTreeNodePage& node = loadNode(page_id);
        const uint16_t entry_count = node.getEntryCount();

        if (node.isLeaf()) {
            for (uint16_t i = 0; i < entry_count; ++i) {
                const float* lower = nullptr;
                const float* upper = nullptr;
                uint64_t value = 0;
                node.getEntryView(i, lower, upper, value);
                ++local_metrics.entries_examined;

                if (pointEqualsEntryPoint(point, lower, upper, point.size())) {
                    matches.push_back(value);
                }
            }
            continue;
        }

        for (uint16_t i = 0; i < entry_count; ++i) {
            const float* lower = nullptr;
            const float* upper = nullptr;
            uint64_t value = 0;
            node.getEntryView(i, lower, upper, value);
            ++local_metrics.entries_examined;

            if (pointInsideBox(point, lower, upper, point.size())) {
                stack.push_back(static_cast<uint32_t>(value));
                ++local_metrics.branches_followed;
            }
        }
    }

    if (metrics != nullptr) {
        *metrics = local_metrics;
    }
    return matches;
}

const RTreeNodePage& RTreeIndex::loadNode(uint32_t page_id) const {
    const auto cached = node_cache_.find(page_id);
    if (cached != node_cache_.end()) {
        return cached->second;
    }

    Page* page = buffer_pool_manager_->fetchPage(page_id);
    if (page == nullptr) {
        throw std::runtime_error("Failed to fetch R-tree node page");
    }

    page->RLock();
    std::vector<uint8_t> raw(page->getData(), page->getData() + PageLayout::kPageSize);
    page->RUnlock();
    buffer_pool_manager_->unpinPage(page_id, false);

    auto [it, inserted] = node_cache_.emplace(page_id, RTreeNodePage(raw));
    if (!inserted) {
        it->second = RTreeNodePage(raw);
    }
    return it->second;
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
    node_cache_.clear();
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
    node_cache_.clear();
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

    // Quadratic split: choose the seed pair that would waste the most space if
    // kept together, then assign remaining entries by largest enlargement delta.
    const std::size_t n = entries.size();
    const std::size_t min_entries = n / 2;

    std::size_t seed_a = 0;
    std::size_t seed_b = 1;
    double best_waste = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const BoundingBox combined = entries[i].mbr.expandedToInclude(entries[j].mbr);
            const double waste = boxMargin(combined) - boxMargin(entries[i].mbr) - boxMargin(entries[j].mbr);
            if (waste > best_waste) {
                best_waste = waste;
                seed_a = i;
                seed_b = j;
            }
        }
    }

    std::vector<bool> assigned(n, false);
    std::vector<RTreeEntry> left_entries;
    std::vector<RTreeEntry> right_entries;
    left_entries.reserve(n);
    right_entries.reserve(n);

    left_entries.push_back(entries[seed_a]);
    right_entries.push_back(entries[seed_b]);
    assigned[seed_a] = true;
    assigned[seed_b] = true;

    BoundingBox left_mbr = entries[seed_a].mbr;
    BoundingBox right_mbr = entries[seed_b].mbr;
    std::size_t assigned_count = 2;

    while (assigned_count < n) {
        const std::size_t remaining = n - assigned_count;

        if (left_entries.size() + remaining == min_entries) {
            for (std::size_t idx = 0; idx < n; ++idx) {
                if (!assigned[idx]) {
                    left_entries.push_back(entries[idx]);
                    left_mbr = left_mbr.expandedToInclude(entries[idx].mbr);
                    assigned[idx] = true;
                    ++assigned_count;
                }
            }
            break;
        }

        if (right_entries.size() + remaining == min_entries) {
            for (std::size_t idx = 0; idx < n; ++idx) {
                if (!assigned[idx]) {
                    right_entries.push_back(entries[idx]);
                    right_mbr = right_mbr.expandedToInclude(entries[idx].mbr);
                    assigned[idx] = true;
                    ++assigned_count;
                }
            }
            break;
        }

        std::size_t chosen = n;
        double max_delta = -1.0;
        double chosen_left_cost = 0.0;
        double chosen_right_cost = 0.0;

        for (std::size_t idx = 0; idx < n; ++idx) {
            if (assigned[idx]) {
                continue;
            }

            const double left_cost = enlargementScore(left_mbr, entries[idx].mbr);
            const double right_cost = enlargementScore(right_mbr, entries[idx].mbr);
            const double delta = std::abs(left_cost - right_cost);
            if (delta > max_delta) {
                max_delta = delta;
                chosen = idx;
                chosen_left_cost = left_cost;
                chosen_right_cost = right_cost;
            }
        }

        if (chosen == n) {
            throw std::runtime_error("Quadratic split failed to choose next entry");
        }

        bool assign_left = false;
        if (chosen_left_cost < chosen_right_cost) {
            assign_left = true;
        } else if (chosen_right_cost < chosen_left_cost) {
            assign_left = false;
        } else {
            const double left_margin = boxMargin(left_mbr);
            const double right_margin = boxMargin(right_mbr);
            if (left_margin < right_margin) {
                assign_left = true;
            } else if (right_margin < left_margin) {
                assign_left = false;
            } else {
                assign_left = left_entries.size() <= right_entries.size();
            }
        }

        if (assign_left) {
            left_entries.push_back(entries[chosen]);
            left_mbr = left_mbr.expandedToInclude(entries[chosen].mbr);
        } else {
            right_entries.push_back(entries[chosen]);
            right_mbr = right_mbr.expandedToInclude(entries[chosen].mbr);
        }

        assigned[chosen] = true;
        ++assigned_count;
    }

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

    // Use log-space enlargement ratio so high-dimensional volumes (128D) do not
    // underflow to 0.0 and collapse all choices to the first child.
    std::size_t best_index = 0;
    double best_log_ratio  = entries[0].mbr.logEnlargementRatio(target);
    double best_log_vol    = entries[0].mbr.logHyperVolume();

    for (std::size_t index = 1; index < entries.size(); ++index) {
        const double log_ratio = entries[index].mbr.logEnlargementRatio(target);
        const double log_vol   = entries[index].mbr.logHyperVolume();
        if (log_ratio < best_log_ratio ||
            (std::abs(log_ratio - best_log_ratio) < 1e-9 && log_vol < best_log_vol)) {
            best_index    = index;
            best_log_ratio = log_ratio;
            best_log_vol   = log_vol;
        }
    }

    return best_index;
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
