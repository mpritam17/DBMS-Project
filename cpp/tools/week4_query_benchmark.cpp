#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "slotted_page.h"
#include "storage_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kSlottedPageMagic = 0x50414745;  // "PAGE"
constexpr uint32_t kRTreeMetaMagic = 0x52544958;    // 'RTIX'
constexpr char kKdCacheMagic[8] = {'K', 'D', 'C', 'A', 'C', 'H', 'E', '1'};
constexpr uint32_t kKdCacheVersion = 1;

struct StoredVector {
    uint64_t id;
    std::vector<float> values;
};

struct QueryMetrics {
    uint64_t query_id;
    long long rtree_us;
    long long kd_us;
    long long brute_us;
    long long rtree_point_us;
    std::size_t point_matches;
    std::size_t point_nodes_visited;
    std::size_t point_entries_examined;
    std::size_t kd_hits;
    std::size_t kd_denom;
    double kd_recall;
    std::size_t hits;
    std::size_t denom;
    double recall;
};

struct BenchmarkOptions {
    bool fair_mode = false;
    bool point_only = false;
    std::size_t bpm_pages = 2048;
    bool bpm_pages_explicit = false;
};

constexpr std::size_t kDefaultBpmPages = 2048;
constexpr std::size_t kAutoBpmMinPages = 2048;
constexpr std::size_t kAutoBpmMaxPages = 8192;
constexpr std::size_t kAutoBpmHeadroomPages = 128;

double l2DistanceSq(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("L2 distance dimension mismatch");
    }
    double dist_sq = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        dist_sq += d * d;
    }
    return dist_sq;
}

float l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    return static_cast<float>(std::sqrt(l2DistanceSq(a, b)));
}

struct KDSearchMetrics {
    std::size_t nodes_visited = 0;
    std::size_t far_branch_checks = 0;
};

class KDTree {
public:
    struct SerializedNode {
        uint32_t point_index = 0;
        uint32_t split_axis = 0;
        int32_t left_child = -1;
        int32_t right_child = -1;
    };

    KDTree(const std::vector<StoredVector>& vectors, std::size_t dimensions)
        : dimensions_(dimensions) {
        initializePointRefs(vectors);

        std::vector<std::size_t> point_indices(points_.size());
        std::iota(point_indices.begin(), point_indices.end(), 0);
        nodes_.reserve(points_.size());
        root_index_ = buildRecursive(point_indices, 0, point_indices.size(), 0);
    }

    KDTree(
        const std::vector<StoredVector>& vectors,
        std::size_t dimensions,
        const std::vector<SerializedNode>& serialized_nodes,
        int root_index)
        : dimensions_(dimensions), root_index_(root_index) {
        initializePointRefs(vectors);

        nodes_.reserve(serialized_nodes.size());
        const int max_index = static_cast<int>(serialized_nodes.size()) - 1;
        for (const auto& node : serialized_nodes) {
            if (node.point_index >= points_.size()) {
                throw std::runtime_error("KD cache node references out-of-range point index");
            }
            if (node.split_axis >= dimensions_) {
                throw std::runtime_error("KD cache node has invalid split axis");
            }
            if (node.left_child < -1 || node.left_child > max_index
                || node.right_child < -1 || node.right_child > max_index) {
                throw std::runtime_error("KD cache node has out-of-range child index");
            }

            nodes_.push_back({
                static_cast<std::size_t>(node.point_index),
                static_cast<std::size_t>(node.split_axis),
                static_cast<int>(node.left_child),
                static_cast<int>(node.right_child),
            });
        }

        if (nodes_.empty()) {
            if (!points_.empty()) {
                throw std::runtime_error("KD cache has no nodes for non-empty dataset");
            }
            if (root_index_ != -1) {
                throw std::runtime_error("KD cache root index must be -1 for empty dataset");
            }
            return;
        }

        const int rebuilt_max_index = static_cast<int>(nodes_.size()) - 1;
        if (root_index_ < 0 || root_index_ > rebuilt_max_index) {
            throw std::runtime_error("KD cache root index is out of range");
        }
    }

    int getRootIndex() const {
        return root_index_;
    }

    std::vector<SerializedNode> exportNodes() const {
        std::vector<SerializedNode> out;
        out.reserve(nodes_.size());
        for (const auto& node : nodes_) {
            out.push_back({
                static_cast<uint32_t>(node.point_index),
                static_cast<uint32_t>(node.split_axis),
                static_cast<int32_t>(node.left_child),
                static_cast<int32_t>(node.right_child),
            });
        }
        return out;
    }

    std::vector<std::pair<float, uint64_t>> searchKNN(
        const std::vector<float>& query,
        std::size_t k,
        KDSearchMetrics* metrics = nullptr) const {
        if (k == 0) {
            return {};
        }
        if (query.size() != dimensions_) {
            throw std::invalid_argument("KD-tree query dimensions do not match tree dimensions");
        }
        if (root_index_ < 0) {
            return {};
        }

        KDSearchMetrics local_metrics{};
        using HeapEntry = std::pair<double, uint64_t>;
        std::priority_queue<HeapEntry> best;

        const std::function<void(int)> dfs = [&](int node_index) {
            if (node_index < 0) {
                return;
            }

            ++local_metrics.nodes_visited;
            const KDNode& node = nodes_[static_cast<std::size_t>(node_index)];
            const KDPointRef& point = points_[node.point_index];

            const double dist_sq = l2DistanceSq(query, *point.coordinates);
            if (best.size() < k || dist_sq < best.top().first) {
                best.push({dist_sq, point.value});
                if (best.size() > k) {
                    best.pop();
                }
            }

            const double delta = static_cast<double>(query[node.split_axis])
                               - static_cast<double>((*point.coordinates)[node.split_axis]);
            const int near_child = delta <= 0.0 ? node.left_child : node.right_child;
            const int far_child = delta <= 0.0 ? node.right_child : node.left_child;

            dfs(near_child);

            ++local_metrics.far_branch_checks;
            const double plane_dist_sq = delta * delta;
            if (far_child >= 0 && (best.size() < k || plane_dist_sq < best.top().first)) {
                dfs(far_child);
            }
        };

        dfs(root_index_);

        if (metrics != nullptr) {
            *metrics = local_metrics;
        }

        std::vector<std::pair<float, uint64_t>> result;
        result.reserve(best.size());
        while (!best.empty()) {
            result.push_back({static_cast<float>(std::sqrt(best.top().first)), best.top().second});
            best.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

private:
    struct KDPointRef {
        uint64_t value = 0;
        const std::vector<float>* coordinates = nullptr;
    };

    struct KDNode {
        std::size_t point_index = 0;
        std::size_t split_axis = 0;
        int left_child = -1;
        int right_child = -1;
    };

    void initializePointRefs(const std::vector<StoredVector>& vectors) {
        if (dimensions_ == 0) {
            throw std::invalid_argument("KD-tree requires dimensions >= 1");
        }

        points_.reserve(vectors.size());
        for (const auto& vec : vectors) {
            points_.push_back({vec.id, &vec.values});
        }
    }

    int buildRecursive(
        std::vector<std::size_t>& point_indices,
        std::size_t begin,
        std::size_t end,
        std::size_t depth) {
        if (begin >= end) {
            return -1;
        }

        const std::size_t axis = depth % dimensions_;
        const std::size_t mid = begin + ((end - begin) / 2);
        std::nth_element(
            point_indices.begin() + static_cast<std::ptrdiff_t>(begin),
            point_indices.begin() + static_cast<std::ptrdiff_t>(mid),
            point_indices.begin() + static_cast<std::ptrdiff_t>(end),
            [&](std::size_t lhs, std::size_t rhs) {
                return (*points_[lhs].coordinates)[axis] < (*points_[rhs].coordinates)[axis];
            });

        const std::size_t point_idx = point_indices[mid];
        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back({
            point_idx,
            axis,
            -1,
            -1,
        });

        nodes_[static_cast<std::size_t>(node_index)].left_child =
            buildRecursive(point_indices, begin, mid, depth + 1);
        nodes_[static_cast<std::size_t>(node_index)].right_child =
            buildRecursive(point_indices, mid + 1, end, depth + 1);

        return node_index;
    }

    std::size_t dimensions_;
    std::vector<KDPointRef> points_;
    std::vector<KDNode> nodes_;
    int root_index_ = -1;
};

std::vector<StoredVector> loadVectorsFromEmbeddingStore(StorageManager& storage, uint16_t* out_dims) {
    std::vector<StoredVector> vectors;
    uint16_t dims = 0;

    for (uint64_t page_id = 0; page_id < storage.pageCount(); ++page_id) {
        const std::vector<uint8_t> raw = storage.readPage(page_id);
        SlottedPage page(raw);
        const PageHeader* header = page.getHeader();

        if (header->magic != kSlottedPageMagic || header->page_type != 0) {
            continue;
        }

        for (uint32_t item_index = 0; item_index < header->item_count; ++item_index) {
            const auto [item_data, item_size] = page.getItem(item_index);
            if (item_size < sizeof(uint64_t) || (item_size - sizeof(uint64_t)) % sizeof(float) != 0) {
                continue;
            }

            const uint16_t item_dims = static_cast<uint16_t>((item_size - sizeof(uint64_t)) / sizeof(float));
            if (item_dims == 0) {
                continue;
            }
            if (dims == 0) {
                dims = item_dims;
            }
            if (item_dims != dims) {
                continue;
            }

            StoredVector vec{};
            std::memcpy(&vec.id, item_data, sizeof(uint64_t));
            vec.values.resize(item_dims);
            std::memcpy(vec.values.data(), item_data + sizeof(uint64_t), item_dims * sizeof(float));
            vectors.push_back(std::move(vec));
        }
    }

    if (out_dims != nullptr) {
        *out_dims = dims;
    }
    return vectors;
}

std::vector<std::pair<float, uint64_t>> bruteForceKNN(
    const std::vector<StoredVector>& vectors,
    const std::vector<float>& query,
    std::size_t k) {
    std::vector<std::pair<float, uint64_t>> results;
    results.reserve(vectors.size());

    for (const auto& vec : vectors) {
        results.push_back({l2Distance(query, vec.values), vec.id});
    }

    const std::size_t limit = std::min(k, results.size());
    std::partial_sort(
        results.begin(),
        results.begin() + limit,
        results.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; });
    results.resize(limit);
    return results;
}

std::size_t recallAtK(
    const std::vector<std::pair<float, uint64_t>>& approx,
    const std::vector<std::pair<float, uint64_t>>& exact) {
    std::unordered_map<uint64_t, bool> truth;
    truth.reserve(exact.size());
    for (const auto& p : exact) {
        truth[p.second] = true;
    }

    std::size_t hits = 0;
    for (const auto& p : approx) {
        if (truth.find(p.second) != truth.end()) {
            ++hits;
        }
    }
    return hits;
}

std::size_t computeAutoBpmPages(std::uint64_t index_page_count) {
    const std::size_t target = static_cast<std::size_t>(index_page_count) + kAutoBpmHeadroomPages;
    return std::max(kAutoBpmMinPages, std::min(kAutoBpmMaxPages, target));
}

template <typename T>
void writeBinary(std::ofstream& out, const T& value, const char* field_name) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!out) {
        throw std::runtime_error(std::string("Failed to write KD cache field: ") + field_name);
    }
}

template <typename T>
T readBinary(std::ifstream& in, const char* field_name) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!in) {
        throw std::runtime_error(std::string("Failed to read KD cache field: ") + field_name);
    }
    return value;
}

void fnv1aUpdate(std::uint64_t* hash, const void* data, std::size_t size_bytes) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < size_bytes; ++i) {
        *hash ^= static_cast<std::uint64_t>(bytes[i]);
        *hash *= 1099511628211ULL;
    }
}

std::uint64_t computeVectorFingerprint(const std::vector<StoredVector>& vectors, uint16_t expected_dims) {
    std::uint64_t hash = 1469598103934665603ULL;
    const std::uint64_t vector_count = static_cast<std::uint64_t>(vectors.size());
    fnv1aUpdate(&hash, &expected_dims, sizeof(expected_dims));
    fnv1aUpdate(&hash, &vector_count, sizeof(vector_count));

    for (const auto& vec : vectors) {
        if (vec.values.size() != expected_dims) {
            throw std::runtime_error("Vector dimensions changed while computing KD fingerprint");
        }
        fnv1aUpdate(&hash, &vec.id, sizeof(vec.id));
        if (!vec.values.empty()) {
            fnv1aUpdate(&hash, vec.values.data(), vec.values.size() * sizeof(float));
        }
    }

    return hash;
}

bool loadPersistentKDTree(
    const std::string& cache_file,
    const std::vector<StoredVector>& vectors,
    uint16_t dims,
    std::uint64_t expected_fingerprint,
    std::unique_ptr<KDTree>* out_tree,
    std::string* out_status) {
    if (out_tree == nullptr) {
        throw std::invalid_argument("loadPersistentKDTree requires a non-null out_tree");
    }
    out_tree->reset();

    if (!std::filesystem::exists(cache_file)) {
        if (out_status != nullptr) {
            *out_status = "missing";
        }
        return false;
    }

    try {
        std::ifstream in(cache_file, std::ios::binary);
        if (!in.is_open()) {
            if (out_status != nullptr) {
                *out_status = "open-failed";
            }
            return false;
        }

        char magic[sizeof(kKdCacheMagic)] = {};
        in.read(magic, static_cast<std::streamsize>(sizeof(magic)));
        if (!in || std::memcmp(magic, kKdCacheMagic, sizeof(magic)) != 0) {
            if (out_status != nullptr) {
                *out_status = "invalid-magic";
            }
            return false;
        }

        const uint32_t version = readBinary<uint32_t>(in, "version");
        if (version != kKdCacheVersion) {
            if (out_status != nullptr) {
                *out_status = "stale-version";
            }
            return false;
        }

        const uint16_t cache_dims = readBinary<uint16_t>(in, "dims");
        const std::uint64_t cache_vector_count = readBinary<std::uint64_t>(in, "vector_count");
        const std::uint64_t cache_fingerprint = readBinary<std::uint64_t>(in, "fingerprint");
        const int32_t cache_root_index = readBinary<int32_t>(in, "root_index");
        const std::uint64_t cache_node_count = readBinary<std::uint64_t>(in, "node_count");

        if (cache_dims != dims) {
            if (out_status != nullptr) {
                *out_status = "stale-dims";
            }
            return false;
        }

        if (cache_vector_count != static_cast<std::uint64_t>(vectors.size())) {
            if (out_status != nullptr) {
                *out_status = "stale-count";
            }
            return false;
        }

        if (cache_fingerprint != expected_fingerprint) {
            if (out_status != nullptr) {
                *out_status = "stale-fingerprint";
            }
            return false;
        }

        if (cache_node_count > static_cast<std::uint64_t>(vectors.size())) {
            if (out_status != nullptr) {
                *out_status = "invalid-node-count";
            }
            return false;
        }

        std::vector<KDTree::SerializedNode> nodes;
        nodes.reserve(static_cast<std::size_t>(cache_node_count));
        for (std::uint64_t i = 0; i < cache_node_count; ++i) {
            const uint32_t point_index = readBinary<uint32_t>(in, "node.point_index");
            const uint32_t split_axis = readBinary<uint32_t>(in, "node.split_axis");
            const int32_t left_child = readBinary<int32_t>(in, "node.left_child");
            const int32_t right_child = readBinary<int32_t>(in, "node.right_child");
            nodes.push_back({point_index, split_axis, left_child, right_child});
        }

        *out_tree = std::make_unique<KDTree>(vectors, dims, nodes, static_cast<int>(cache_root_index));
        if (out_status != nullptr) {
            *out_status = "loaded";
        }
        return true;
    } catch (...) {
        if (out_status != nullptr) {
            *out_status = "invalid";
        }
        return false;
    }
}

bool savePersistentKDTree(
    const std::string& cache_file,
    const KDTree& tree,
    const std::vector<StoredVector>& vectors,
    uint16_t dims,
    std::uint64_t fingerprint) {
    const std::filesystem::path target_path(cache_file);
    const std::filesystem::path temp_path = target_path.string() + ".tmp";

    try {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        out.write(kKdCacheMagic, static_cast<std::streamsize>(sizeof(kKdCacheMagic)));
        if (!out) {
            return false;
        }

        const auto nodes = tree.exportNodes();
        writeBinary<uint32_t>(out, kKdCacheVersion, "version");
        writeBinary<uint16_t>(out, dims, "dims");
        writeBinary<std::uint64_t>(out, static_cast<std::uint64_t>(vectors.size()), "vector_count");
        writeBinary<std::uint64_t>(out, fingerprint, "fingerprint");
        writeBinary<int32_t>(out, static_cast<int32_t>(tree.getRootIndex()), "root_index");
        writeBinary<std::uint64_t>(out, static_cast<std::uint64_t>(nodes.size()), "node_count");

        for (const auto& node : nodes) {
            writeBinary<uint32_t>(out, node.point_index, "node.point_index");
            writeBinary<uint32_t>(out, node.split_axis, "node.split_axis");
            writeBinary<int32_t>(out, node.left_child, "node.left_child");
            writeBinary<int32_t>(out, node.right_child, "node.right_child");
        }

        out.flush();
        out.close();

        std::error_code ec;
        std::filesystem::remove(target_path, ec);
        ec.clear();
        std::filesystem::rename(temp_path, target_path, ec);
        if (ec) {
            std::filesystem::remove(temp_path, ec);
            return false;
        }

        return true;
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        return false;
    }
}

void warmIndexCache(BufferPoolManager& bpm, std::uint64_t page_count) {
    for (std::uint64_t page_id = 0; page_id < page_count; ++page_id) {
        Page* page = bpm.fetchPage(static_cast<uint32_t>(page_id));
        if (page == nullptr) {
            throw std::runtime_error("Failed to warm index cache: fetchPage returned null");
        }
        if (!bpm.unpinPage(static_cast<uint32_t>(page_id), false)) {
            throw std::runtime_error("Failed to warm index cache: unpinPage failed");
        }
    }
}

bool parseAllSelector(const std::string& selector, std::size_t* out_limit) {
    if (selector == "all") {
        if (out_limit != nullptr) {
            *out_limit = 0;
        }
        return true;
    }

    const std::string prefix = "all:";
    if (selector.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string limit_str = selector.substr(prefix.size());
    if (limit_str.empty()) {
        throw std::runtime_error("Invalid query selector 'all:'; expected all:<positive_integer>");
    }

    const std::size_t limit = static_cast<std::size_t>(std::stoull(limit_str));
    if (limit == 0) {
        throw std::runtime_error("Invalid query selector; all:<N> requires N >= 1");
    }

    if (out_limit != nullptr) {
        *out_limit = limit;
    }
    return true;
}

bool parseVecSelector(const std::string& selector, std::vector<float>& out_vec) {
    const std::string prefix = "vec:";
    if (selector.rfind(prefix, 0) != 0) {
        return false;
    }
    
    std::string vals = selector.substr(prefix.size());
    std::size_t start = 0, end = 0;
    while ((end = vals.find(',', start)) != std::string::npos) {
        out_vec.push_back(std::stof(vals.substr(start, end - start)));
        start = end + 1;
    }
    out_vec.push_back(std::stof(vals.substr(start)));
    return true;
}

bool isReusableTempIndex(const std::string& index_db_file, uint16_t expected_dims) {
    if (!std::filesystem::exists(index_db_file)) {
        return false;
    }

    try {
        StorageManager storage;
        storage.open(index_db_file);
        if (storage.pageCount() == 0) {
            return false;
        }

        const std::vector<uint8_t> page0 = storage.readPage(0);
        if (page0.size() < PageLayout::kHeaderSize + 14) {
            return false;
        }

        const PageHeader* ph = reinterpret_cast<const PageHeader*>(page0.data());
        if (ph->magic != RTREE_PAGE_MAGIC) {
            return false;
        }

        uint32_t meta_magic = 0;
        std::memcpy(&meta_magic, page0.data() + sizeof(PageHeader), sizeof(uint32_t));
        if (meta_magic != kRTreeMetaMagic) {
            return false;
        }

        uint16_t meta_dims = 0;
        std::memcpy(
            &meta_dims,
            page0.data() + sizeof(PageHeader) + (3 * sizeof(uint32_t)),
            sizeof(uint16_t));

        return meta_dims == expected_dims;
    } catch (...) {
        return false;
    }
}

QueryMetrics benchmarkSingleQueryRaw(
    const RTreeIndex& index,
    const KDTree& kd_tree,
    const std::vector<StoredVector>& unique_vectors,
    const std::vector<float>& query,
    uint64_t query_id_label,
    std::size_t k,
    std::vector<std::pair<float, uint64_t>>* out_results = nullptr,
    std::vector<uint64_t>* out_point_matches = nullptr) {
    const auto rtree_start = std::chrono::high_resolution_clock::now();
    const auto rtree_results = index.searchKNN(query, k);
    const auto rtree_end = std::chrono::high_resolution_clock::now();

    const auto kd_start = std::chrono::high_resolution_clock::now();
    const auto kd_results = kd_tree.searchKNN(query, k);
    const auto kd_end = std::chrono::high_resolution_clock::now();

    const auto brute_start = std::chrono::high_resolution_clock::now();
    const auto brute_results = bruteForceKNN(unique_vectors, query, k);
    const auto brute_end = std::chrono::high_resolution_clock::now();

    RTreeIndex::PointSearchMetrics point_metrics{};
    const auto point_start = std::chrono::high_resolution_clock::now();
    const auto point_matches = index.searchPoint(query, &point_metrics);
    const auto point_end = std::chrono::high_resolution_clock::now();

    const auto rtree_us = std::chrono::duration_cast<std::chrono::microseconds>(rtree_end - rtree_start).count();
    const auto kd_us = std::chrono::duration_cast<std::chrono::microseconds>(kd_end - kd_start).count();
    const auto brute_us = std::chrono::duration_cast<std::chrono::microseconds>(brute_end - brute_start).count();
    const auto point_us = std::chrono::duration_cast<std::chrono::microseconds>(point_end - point_start).count();

    const std::size_t hits = recallAtK(rtree_results, brute_results);
    const std::size_t denom = std::min(rtree_results.size(), brute_results.size());
    const double recall = denom == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(denom);
    const std::size_t kd_hits = recallAtK(kd_results, brute_results);
    const std::size_t kd_denom = std::min(kd_results.size(), brute_results.size());
    const double kd_recall = kd_denom == 0 ? 0.0 : static_cast<double>(kd_hits) / static_cast<double>(kd_denom);

    if (out_results) {
        *out_results = rtree_results;
    }
    if (out_point_matches) {
        *out_point_matches = point_matches;
    }

    return {
        query_id_label,
        rtree_us,
        kd_us,
        brute_us,
        point_us,
        point_matches.size(),
        point_metrics.nodes_visited,
        point_metrics.entries_examined,
        kd_hits,
        kd_denom,
        kd_recall,
        hits,
        denom,
        recall,
    };
}

QueryMetrics benchmarkSingleQuery(
    const RTreeIndex& index,
    const KDTree& kd_tree,
    const std::vector<StoredVector>& unique_vectors,
    const std::unordered_map<uint64_t, std::vector<float>>& by_id,
    uint64_t query_id,
    std::size_t k) {
    const auto query_it = by_id.find(query_id);
    if (query_it == by_id.end()) {
        throw std::runtime_error("query_id not found in embedding store");
    }
    return benchmarkSingleQueryRaw(index, kd_tree, unique_vectors, query_it->second, query_id, k);
}

QueryMetrics benchmarkPointOnlyRaw(
    const RTreeIndex& index,
    const std::vector<float>& query,
    uint64_t query_id_label,
    std::vector<uint64_t>* out_point_matches = nullptr) {
    RTreeIndex::PointSearchMetrics point_metrics{};
    const auto point_start = std::chrono::high_resolution_clock::now();
    const auto point_matches = index.searchPoint(query, &point_metrics);
    const auto point_end = std::chrono::high_resolution_clock::now();
    const auto point_us = std::chrono::duration_cast<std::chrono::microseconds>(point_end - point_start).count();

    if (out_point_matches) {
        *out_point_matches = point_matches;
    }

    return {
        query_id_label,
        0,
        0,
        0,
        point_us,
        point_matches.size(),
        point_metrics.nodes_visited,
        point_metrics.entries_examined,
        0,
        0,
        0.0,
        0,
        0,
        0.0,
    };
}

QueryMetrics benchmarkPointOnlyQuery(
    const RTreeIndex& index,
    const std::unordered_map<uint64_t, std::vector<float>>& by_id,
    uint64_t query_id) {
    const auto query_it = by_id.find(query_id);
    if (query_it == by_id.end()) {
        throw std::runtime_error("query_id not found in embedding store");
    }
    return benchmarkPointOnlyRaw(index, query_it->second, query_id);
}

void writeCsv(const std::string& csv_path, const std::vector<QueryMetrics>& rows) {
    std::ofstream out(csv_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open CSV output path: " + csv_path);
    }

    out << "query_id,rtree_us,kd_us,brute_us,rtree_point_us,point_matches,point_nodes_visited,point_entries_examined,kd_hits,kd_denom,kd_recall,hits,denom,recall\n";
    for (const auto& row : rows) {
        out << row.query_id << ","
            << row.rtree_us << ","
            << row.kd_us << ","
            << row.brute_us << ","
            << row.rtree_point_us << ","
            << row.point_matches << ","
            << row.point_nodes_visited << ","
            << row.point_entries_examined << ","
            << row.kd_hits << ","
            << row.kd_denom << ","
            << row.kd_recall << ","
            << row.hits << ","
            << row.denom << ","
            << row.recall << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <db_file> <query_id|all|all:N|vec:x,y,...> <k>"
                  << " [csv_output_path] [--fair] [--point-only] [--bpm-pages N|auto]\n";
        return 1;
    }

    const std::string db_file = argv[1];
    const std::string query_selector = argv[2];
    const std::size_t k = static_cast<std::size_t>(std::stoull(argv[3]));
    BenchmarkOptions options{};
    options.bpm_pages = kDefaultBpmPages;
    bool emit_csv = false;
    std::string csv_path;

    int argi = 4;
    if (argi < argc) {
        const std::string maybe_csv = argv[argi];
        if (maybe_csv.rfind("--", 0) != 0) {
            emit_csv = true;
            csv_path = maybe_csv;
            ++argi;
        }
    }

    while (argi < argc) {
        const std::string arg = argv[argi];
        if (arg == "--fair") {
            options.fair_mode = true;
            ++argi;
            continue;
        }
        if (arg == "--point-only") {
            options.point_only = true;
            ++argi;
            continue;
        }
        if (arg == "--bpm-pages") {
            if (argi + 1 >= argc) {
                throw std::runtime_error("--bpm-pages requires a value (N or auto)");
            }
            const std::string value = argv[argi + 1];
            if (value == "auto") {
                options.bpm_pages_explicit = false;
            } else {
                const std::size_t parsed = static_cast<std::size_t>(std::stoull(value));
                if (parsed == 0) {
                    throw std::runtime_error("--bpm-pages requires N >= 1");
                }
                options.bpm_pages = parsed;
                options.bpm_pages_explicit = true;
            }
            argi += 2;
            continue;
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (k == 0) {
        std::cerr << "k must be >= 1\n";
        return 1;
    }

    try {
        uint16_t dims = 0;
        std::vector<StoredVector> vectors;
        {
            // Read vectors from the embedding store only.
            StorageManager storage;
            storage.open(db_file);
            vectors = loadVectorsFromEmbeddingStore(storage, &dims);
        }
        if (vectors.empty() || dims == 0) {
            throw std::runtime_error("No vectors found in slotted-page embedding store");
        }

        if (query_selector == "dim") {
            std::cout << "dims: " << dims << "\n";
            return 0;
        }

        std::unordered_map<uint64_t, std::vector<float>> by_id;
        by_id.reserve(vectors.size());
        for (const auto& v : vectors) {
            by_id[v.id] = v.values;
        }

        std::vector<StoredVector> unique_vectors;
        unique_vectors.reserve(by_id.size());
        for (const auto& kv : by_id) {
            unique_vectors.push_back({kv.first, kv.second});
        }
        std::sort(unique_vectors.begin(), unique_vectors.end(), [](const StoredVector& a, const StoredVector& b) {
            return a.id < b.id;
        });

        const std::string kd_cache_file = db_file + ".kd_tmp.bin";
        std::string kd_cache_status = options.point_only ? "point-only-skip" : "missing";
        std::unique_ptr<KDTree> kd_tree;
        long long kd_build_us = 0;
        if (!options.point_only) {
            const std::uint64_t kd_fingerprint = computeVectorFingerprint(unique_vectors, dims);
            if (!loadPersistentKDTree(
                    kd_cache_file,
                    unique_vectors,
                    dims,
                    kd_fingerprint,
                    &kd_tree,
                    &kd_cache_status)) {
                const auto kd_build_start = std::chrono::high_resolution_clock::now();
                kd_tree = std::make_unique<KDTree>(unique_vectors, dims);
                const auto kd_build_end = std::chrono::high_resolution_clock::now();
                kd_build_us = std::chrono::duration_cast<std::chrono::microseconds>(kd_build_end - kd_build_start).count();

                const bool cache_saved = savePersistentKDTree(
                    kd_cache_file,
                    *kd_tree,
                    unique_vectors,
                    dims,
                    kd_fingerprint);
                if (cache_saved) {
                    kd_cache_status = kd_cache_status == "missing" ? "built-saved" : "rebuilt-saved";
                } else {
                    kd_cache_status = kd_cache_status == "missing" ? "built-unsaved" : "rebuilt-unsaved";
                }
            }
        }

        std::vector<uint64_t> query_ids;
        std::size_t all_limit = 0;
        std::vector<float> parsed_vec_query;
        bool is_custom_vec = parseVecSelector(query_selector, parsed_vec_query);

        if (is_custom_vec) {
            query_ids.push_back(999999); // dummy ID
            if (parsed_vec_query.size() != dims) {
                throw std::runtime_error("Custom vector dimension mismatch");
            }
        } else if (parseAllSelector(query_selector, &all_limit)) {
            query_ids.reserve(unique_vectors.size());
            for (const auto& v : unique_vectors) {
                query_ids.push_back(v.id);
            }
            if (all_limit > 0 && query_ids.size() > all_limit) {
                query_ids.resize(all_limit);
            }
        } else {
            query_ids.push_back(std::stoull(query_selector));
        }

        // Build the R-tree in a separate temp DB to avoid mixing index pages
        // with embedding slotted pages in the source file. Re-use if exists!
        const std::string index_db_file = db_file + ".rtree_tmp.db";
        bool build_index = true;
        if (std::filesystem::exists(index_db_file)) {
            if (isReusableTempIndex(index_db_file, dims)) {
                build_index = false;
            } else {
                std::cout << "Existing temp R-tree index is stale/corrupt for this dataset; rebuilding.\n";
                std::filesystem::remove(index_db_file);
            }
        }
        
        StorageManager index_storage;
        index_storage.open(index_db_file);

        const std::uint64_t index_page_count = index_storage.pageCount();
        if (!options.bpm_pages_explicit) {
            options.bpm_pages = computeAutoBpmPages(index_page_count);
        }

        BufferPoolManager bpm(options.bpm_pages, &index_storage);
        
        RTreeIndex* index_ptr = nullptr;
        if (build_index) {
            index_ptr = new RTreeIndex(&bpm, dims);
            std::cout << "Building R-Tree index... (this may take a moment)\n";
            for (const auto& v : unique_vectors) {
                index_ptr->insertPoint(v.values, v.id);
            }
            bpm.flushAllPages();
            std::cout << "R-Tree index built.\n";
        } else {
            index_ptr = new RTreeIndex(&bpm, static_cast<uint32_t>(0));
            std::cout << "Loaded existing R-Tree index from " << index_db_file << "\n";
        }
        RTreeIndex& index = *index_ptr;
        const std::uint64_t measured_index_page_count = index_storage.pageCount();

        if (options.fair_mode) {
            warmIndexCache(bpm, measured_index_page_count);

            // Prime traversal and branch predictor/cache effects before timed run.
            if (is_custom_vec) {
                if (!options.point_only) {
                    (void) index.searchKNN(parsed_vec_query, k);
                    (void) kd_tree->searchKNN(parsed_vec_query, k);
                }
                (void) index.searchPoint(parsed_vec_query);
            } else {
                for (uint64_t qid : query_ids) {
                    const auto query_it = by_id.find(qid);
                    if (query_it != by_id.end()) {
                        if (!options.point_only) {
                            (void) index.searchKNN(query_it->second, k);
                            (void) kd_tree->searchKNN(query_it->second, k);
                        }
                        (void) index.searchPoint(query_it->second);
                    }
                }
            }
        }

        std::vector<std::pair<float, uint64_t>> out_results;
        std::vector<uint64_t> out_point_matches;
        std::vector<QueryMetrics> all_metrics;
        all_metrics.reserve(query_ids.size());

        if (options.point_only) {
            if (is_custom_vec) {
                all_metrics.push_back(benchmarkPointOnlyRaw(
                    index,
                    parsed_vec_query,
                    999999,
                    &out_point_matches));
            } else {
                for (uint64_t qid : query_ids) {
                    all_metrics.push_back(benchmarkPointOnlyQuery(index, by_id, qid));
                }
            }
        } else {
            if (is_custom_vec) {
                all_metrics.push_back(benchmarkSingleQueryRaw(
                    index,
                    *kd_tree,
                    unique_vectors,
                    parsed_vec_query,
                    999999,
                    k,
                    &out_results,
                    &out_point_matches));
            } else {
                for (uint64_t qid : query_ids) {
                    all_metrics.push_back(benchmarkSingleQuery(index, *kd_tree, unique_vectors, by_id, qid, k));
                }
            }
        }

        long long total_rtree_us = 0;
        long long total_kd_us = 0;
        long long total_brute_us = 0;
        long long total_point_us = 0;
        std::size_t total_point_matches = 0;
        std::size_t total_point_nodes = 0;
        std::size_t total_point_entries = 0;
        std::size_t point_hit_queries = 0;
        std::size_t total_kd_hits = 0;
        std::size_t total_kd_denom = 0;
        std::size_t total_hits = 0;
        std::size_t total_denom = 0;
        for (const auto& m : all_metrics) {
            total_rtree_us += m.rtree_us;
            total_kd_us += m.kd_us;
            total_brute_us += m.brute_us;
            total_point_us += m.rtree_point_us;
            total_point_matches += m.point_matches;
            total_point_nodes += m.point_nodes_visited;
            total_point_entries += m.point_entries_examined;
            if (m.point_matches > 0) {
                ++point_hit_queries;
            }
            total_kd_hits += m.kd_hits;
            total_kd_denom += m.kd_denom;
            total_hits += m.hits;
            total_denom += m.denom;
        }

        const double avg_rtree_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_rtree_us) / static_cast<double>(all_metrics.size());
        const double avg_kd_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_kd_us) / static_cast<double>(all_metrics.size());
        const double avg_brute_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_brute_us) / static_cast<double>(all_metrics.size());
        const double avg_point_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_point_us) / static_cast<double>(all_metrics.size());
        const double point_hit_rate = all_metrics.empty() ? 0.0
            : static_cast<double>(point_hit_queries) / static_cast<double>(all_metrics.size());
        const double point_matches_per_query = all_metrics.empty() ? 0.0
            : static_cast<double>(total_point_matches) / static_cast<double>(all_metrics.size());
        const double avg_point_nodes = all_metrics.empty() ? 0.0
            : static_cast<double>(total_point_nodes) / static_cast<double>(all_metrics.size());
        const double avg_point_entries = all_metrics.empty() ? 0.0
            : static_cast<double>(total_point_entries) / static_cast<double>(all_metrics.size());
        const double avg_kd_recall = total_kd_denom == 0 ? 0.0
            : static_cast<double>(total_kd_hits) / static_cast<double>(total_kd_denom);
        const double avg_recall = total_denom == 0 ? 0.0
            : static_cast<double>(total_hits) / static_cast<double>(total_denom);

        const std::uint64_t bpm_fetch_requests = bpm.getFetchRequests();
        const std::uint64_t bpm_fetch_hits = bpm.getFetchHits();
        const std::uint64_t bpm_fetch_misses = bpm.getFetchMisses();
        const double bpm_hit_rate = bpm.getHitRate();

        std::cout << "Week 4 Query Benchmark\n";
        std::cout << "  vectors(raw): " << vectors.size() << "\n";
        std::cout << "  vectors(unique): " << unique_vectors.size() << "\n";
        std::cout << "  dims: " << dims << "\n";
        std::cout << "  query_selector: " << query_selector << "\n";
        std::cout << "  queries_run: " << all_metrics.size() << "\n";
        std::cout << "  k: " << k << "\n";
        std::cout << "  fair_mode: " << (options.fair_mode ? "on" : "off") << "\n";
        std::cout << "  point_only_mode: " << (options.point_only ? "on" : "off") << "\n";
        std::cout << "  bpm_pages: " << options.bpm_pages << "\n";
        std::cout << "  kd_build_us: " << kd_build_us << "\n";
        std::cout << "  kd_cache_file: " << kd_cache_file << "\n";
        std::cout << "  kd_cache_status: " << kd_cache_status << "\n";
        std::cout << "\n";
        std::cout << "Average R-tree KNN latency: " << avg_rtree_us << " us\n";
        std::cout << "Average KD-tree latency: " << avg_kd_us << " us\n";
        std::cout << "Average brute-force latency: " << avg_brute_us << " us\n";
        std::cout << "Average R-tree point-search latency: " << avg_point_us << " us\n";
        std::cout << "Point-search hit-rate: " << point_hit_rate << " (" << point_hit_queries
                  << "/" << all_metrics.size() << ")\n";
        std::cout << "Point-search matches/query: " << point_matches_per_query << "\n";
        std::cout << "Point-search avg nodes visited: " << avg_point_nodes << "\n";
        std::cout << "Point-search avg entries examined: " << avg_point_entries << "\n";
        std::cout << "Average KD recall@k: " << avg_kd_recall << " ("
                  << total_kd_hits << "/" << total_kd_denom << ")\n";
        std::cout << "Average recall@k: " << avg_recall << " (" << total_hits << "/" << total_denom << ")\n";
        std::cout << "Buffer-pool fetch requests: " << bpm_fetch_requests << "\n";
        std::cout << "Buffer-pool hits: " << bpm_fetch_hits << "\n";
        std::cout << "Buffer-pool misses: " << bpm_fetch_misses << "\n";
        std::cout << "Buffer-pool hit-rate: " << bpm_hit_rate
              << " (" << (bpm_hit_rate * 100.0) << "%)\n";
        std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                  << ", writes=" << StorageManager::disk_writes.load() << "\n";
        std::cout << "RTree metadata page id: " << index.getMetaPageId() << "\n";

        const std::size_t print_n = std::min<std::size_t>(5, all_metrics.size());
        std::cout << "\nFirst " << print_n
                  << " per-query rows (query_id, rtree_us, kd_us, brute_us, point_us, point_matches, kd_recall, recall):\n";
        for (std::size_t i = 0; i < print_n; ++i) {
            std::cout << "  " << i + 1 << ". "
                      << all_metrics[i].query_id << ", "
                      << all_metrics[i].rtree_us << ", "
                      << all_metrics[i].kd_us << ", "
                      << all_metrics[i].brute_us << ", "
                      << all_metrics[i].rtree_point_us << ", "
                      << all_metrics[i].point_matches << ", "
                      << all_metrics[i].kd_recall << ", "
                      << all_metrics[i].recall << "\n";
        }

        if (emit_csv) {
            writeCsv(csv_path, all_metrics);
            std::cout << "CSV written: " << csv_path << "\n";
        }

        if (is_custom_vec) {
            std::cout << "\nCustom Vector Point Matches:\n";
            for (uint64_t point_id : out_point_matches) {
                std::cout << "{ \"point_id\": " << point_id << " }\n";
            }

            if (!options.point_only) {
                std::cout << "\nCustom Vector Neighbors:\n";
                for (const auto& pair : out_results) {
                    std::cout << "{ \"id\": " << pair.second << ", \"distance\": " << pair.first << " }\n";
                }
            }
        }

        delete index_ptr;

        // We will no longer remove the R-tree index. It takes too long to build 
        // every single time. It remains for rapid follow-up queries.
        // std::filesystem::remove(index_db_file);

        // StorageManager closes in its destructor after BPM/index teardown.
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
