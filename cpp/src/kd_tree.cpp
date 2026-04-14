#include "kd_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

KDTree::KDTree(uint16_t dimensions) : dimensions_(dimensions), root_(-1) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("KDTree requires at least one dimension");
    }
}

void KDTree::build(const std::vector<KDEntry>& entries) {
    entries_.clear();
    nodes_.clear();
    root_ = -1;

    entries_ = entries;
    if (entries_.empty()) {
        return;
    }

    for (const auto& entry : entries_) {
        if (entry.values.size() != dimensions_) {
            throw std::invalid_argument("KDTree entry dimensions do not match tree dimensions");
        }
    }

    std::vector<uint32_t> indices(entries_.size());
    for (uint32_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    nodes_.reserve(entries_.size());
    root_ = buildRecursive(indices, 0, static_cast<int>(indices.size()), 0);
}

std::vector<std::pair<float, uint64_t>> KDTree::searchKNN(const std::vector<float>& query, std::size_t k) const {
    if (k == 0 || root_ == -1) {
        return {};
    }

    validateQueryDimensions(query);

    std::priority_queue<std::pair<double, uint64_t>> best;
    searchKNNRecursive(root_, query, k, best);

    std::vector<std::pair<float, uint64_t>> output;
    output.reserve(best.size());
    while (!best.empty()) {
        const auto top = best.top();
        best.pop();
        output.push_back({static_cast<float>(std::sqrt(top.first)), top.second});
    }

    std::reverse(output.begin(), output.end());
    return output;
}

std::vector<uint64_t> KDTree::searchExactPoint(const std::vector<float>& query) const {
    if (root_ == -1) {
        return {};
    }

    validateQueryDimensions(query);

    std::vector<uint64_t> matches;
    searchExactRecursive(root_, query, matches);
    std::sort(matches.begin(), matches.end());
    return matches;
}

std::size_t KDTree::size() const {
    return entries_.size();
}

uint16_t KDTree::dimensions() const {
    return dimensions_;
}

int KDTree::buildRecursive(std::vector<uint32_t>& indices, int begin, int end, uint16_t depth) {
    if (begin >= end) {
        return -1;
    }

    const uint16_t axis = static_cast<uint16_t>(depth % dimensions_);
    const int mid = begin + ((end - begin) / 2);

    auto comp = [this, axis](uint32_t left, uint32_t right) {
        const float lv = entries_[left].values[axis];
        const float rv = entries_[right].values[axis];
        if (lv < rv) {
            return true;
        }
        if (lv > rv) {
            return false;
        }
        return entries_[left].id < entries_[right].id;
    };

    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end, comp);

    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back({indices[mid], axis, -1, -1});

    nodes_[node_index].left = buildRecursive(indices, begin, mid, static_cast<uint16_t>(depth + 1));
    nodes_[node_index].right = buildRecursive(indices, mid + 1, end, static_cast<uint16_t>(depth + 1));
    return node_index;
}

double KDTree::squaredDistance(const std::vector<float>& a, const std::vector<float>& b) const {
    double d2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        d2 += d * d;
    }
    return d2;
}

void KDTree::searchKNNRecursive(
    int node_index,
    const std::vector<float>& query,
    std::size_t k,
    std::priority_queue<std::pair<double, uint64_t>>& best) const {
    if (node_index < 0) {
        return;
    }

    const Node& node = nodes_[static_cast<std::size_t>(node_index)];
    const KDEntry& entry = entries_[node.entry_index];

    const double dist_sq = squaredDistance(query, entry.values);
    if (best.size() < k) {
        best.push({dist_sq, entry.id});
    } else if (dist_sq < best.top().first ||
               (dist_sq == best.top().first && entry.id < best.top().second)) {
        best.pop();
        best.push({dist_sq, entry.id});
    }

    const uint16_t axis = node.axis;
    const double diff = static_cast<double>(query[axis]) - static_cast<double>(entry.values[axis]);

    const int near_child = diff <= 0.0 ? node.left : node.right;
    const int far_child = diff <= 0.0 ? node.right : node.left;

    searchKNNRecursive(near_child, query, k, best);

    const double current_worst = (best.size() < k)
        ? std::numeric_limits<double>::infinity()
        : best.top().first;

    if ((diff * diff) <= current_worst) {
        searchKNNRecursive(far_child, query, k, best);
    }
}

void KDTree::searchExactRecursive(int node_index, const std::vector<float>& query, std::vector<uint64_t>& matches) const {
    if (node_index < 0) {
        return;
    }

    const Node& node = nodes_[static_cast<std::size_t>(node_index)];
    const KDEntry& entry = entries_[node.entry_index];

    if (std::equal(entry.values.begin(), entry.values.end(), query.begin())) {
        matches.push_back(entry.id);
    }

    const uint16_t axis = node.axis;
    const float q = query[axis];
    const float p = entry.values[axis];

    if (q < p) {
        searchExactRecursive(node.left, query, matches);
    } else if (q > p) {
        searchExactRecursive(node.right, query, matches);
    } else {
        searchExactRecursive(node.left, query, matches);
        searchExactRecursive(node.right, query, matches);
    }
}

void KDTree::validateQueryDimensions(const std::vector<float>& query) const {
    if (query.size() != dimensions_) {
        throw std::invalid_argument("KDTree query dimensions do not match tree dimensions");
    }
}
