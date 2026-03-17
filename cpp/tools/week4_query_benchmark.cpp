#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "slotted_page.h"
#include "storage_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kSlottedPageMagic = 0x50414745;  // "PAGE"

struct StoredVector {
    uint64_t id;
    std::vector<float> values;
};

float l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("L2 distance dimension mismatch");
    }
    double dist_sq = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        dist_sq += d * d;
    }
    return static_cast<float>(std::sqrt(dist_sq));
}

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
                throw std::runtime_error("Inconsistent vector dimensions in embedding store");
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <db_file> <query_id> <k>\n";
        return 1;
    }

    const std::string db_file = argv[1];
    const uint64_t query_id = std::stoull(argv[2]);
    const std::size_t k = static_cast<std::size_t>(std::stoull(argv[3]));
    if (k == 0) {
        std::cerr << "k must be >= 1\n";
        return 1;
    }

    try {
        StorageManager storage;
        storage.open(db_file);

        uint16_t dims = 0;
        std::vector<StoredVector> vectors = loadVectorsFromEmbeddingStore(storage, &dims);
        if (vectors.empty() || dims == 0) {
            throw std::runtime_error("No vectors found in slotted-page embedding store");
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

        const auto query_it = by_id.find(query_id);
        if (query_it == by_id.end()) {
            throw std::runtime_error("query_id not found in embedding store");
        }
        const std::vector<float>& query = query_it->second;

        StorageManager::disk_reads = 0;
        StorageManager::disk_writes = 0;

        BufferPoolManager bpm(64, &storage);
        RTreeIndex index(&bpm, dims);

        for (const auto& v : unique_vectors) {
            index.insertPoint(v.values, v.id);
        }
        bpm.flushAllPages();

        const auto rtree_start = std::chrono::high_resolution_clock::now();
        const auto rtree_results = index.searchKNN(query, k);
        const auto rtree_end = std::chrono::high_resolution_clock::now();

        const auto brute_start = std::chrono::high_resolution_clock::now();
        const auto brute_results = bruteForceKNN(unique_vectors, query, k);
        const auto brute_end = std::chrono::high_resolution_clock::now();

        const auto rtree_us = std::chrono::duration_cast<std::chrono::microseconds>(rtree_end - rtree_start).count();
        const auto brute_us = std::chrono::duration_cast<std::chrono::microseconds>(brute_end - brute_start).count();

        const std::size_t hits = recallAtK(rtree_results, brute_results);
        const std::size_t denom = std::min(rtree_results.size(), brute_results.size());
        const double recall = denom == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(denom);

        std::cout << "Week 4 Query Benchmark\n";
        std::cout << "  vectors(raw): " << vectors.size() << "\n";
        std::cout << "  vectors(unique): " << unique_vectors.size() << "\n";
        std::cout << "  dims: " << dims << "\n";
        std::cout << "  query_id: " << query_id << "\n";
        std::cout << "  k: " << k << "\n";
        std::cout << "\n";
        std::cout << "R-tree KNN latency: " << rtree_us << " us\n";
        std::cout << "Brute-force latency: " << brute_us << " us\n";
        std::cout << "Recall@" << denom << ": " << recall << " (" << hits << "/" << denom << ")\n";
        std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                  << ", writes=" << StorageManager::disk_writes.load() << "\n";
        std::cout << "RTree metadata page id: " << index.getMetaPageId() << "\n";

        const std::size_t print_n = std::min<std::size_t>(5, rtree_results.size());
        std::cout << "\nTop " << print_n << " R-tree results (distance, id):\n";
        for (std::size_t i = 0; i < print_n; ++i) {
            std::cout << "  " << i + 1 << ". " << rtree_results[i].first << ", " << rtree_results[i].second << "\n";
        }

        // StorageManager closes in its destructor after BPM/index teardown.
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
