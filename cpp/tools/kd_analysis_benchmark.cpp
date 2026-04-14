#include "buffer_pool_manager.h"
#include "kd_tree.h"
#include "rtree_index.h"
#include "slotted_page.h"
#include "storage_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kSlottedPageMagic = 0x50414745;  // "PAGE"

struct StoredVector {
    uint64_t id;
    std::vector<float> values;
};

struct AnalysisMetrics {
    uint64_t query_id;
    long long rtree_us;
    long long kdtree_us;
    long long brute_us;
    std::size_t rtree_hits;
    std::size_t kdtree_hits;
    std::size_t denom;
    double rtree_recall;
    double kdtree_recall;
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
            const auto item = page.getItem(item_index);
            const uint8_t* item_data = item.first;
            const size_t item_size = item.second;

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
    std::vector<std::pair<float, uint64_t>> rows;
    rows.reserve(vectors.size());

    for (const auto& vec : vectors) {
        rows.push_back({l2Distance(query, vec.values), vec.id});
    }

    const std::size_t limit = std::min(k, rows.size());
    std::partial_sort(
        rows.begin(),
        rows.begin() + limit,
        rows.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; });
    rows.resize(limit);
    return rows;
}

std::size_t hitsAtK(
    const std::vector<std::pair<float, uint64_t>>& approx,
    const std::vector<std::pair<float, uint64_t>>& exact) {
    std::unordered_map<uint64_t, bool> truth;
    truth.reserve(exact.size());
    for (const auto& row : exact) {
        truth[row.second] = true;
    }

    std::size_t hits = 0;
    for (const auto& row : approx) {
        if (truth.find(row.second) != truth.end()) {
            ++hits;
        }
    }
    return hits;
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

    std::string values = selector.substr(prefix.size());
    std::size_t start = 0;
    std::size_t end = 0;
    while ((end = values.find(',', start)) != std::string::npos) {
        out_vec.push_back(std::stof(values.substr(start, end - start)));
        start = end + 1;
    }
    out_vec.push_back(std::stof(values.substr(start)));
    return true;
}

AnalysisMetrics runSingleAnalysisRaw(
    const RTreeIndex& rtree,
    const KDTree& kdtree,
    const std::vector<StoredVector>& vectors,
    const std::vector<float>& query,
    uint64_t query_id_label,
    std::size_t k,
    std::vector<std::pair<float, uint64_t>>* out_rtree = nullptr,
    std::vector<std::pair<float, uint64_t>>* out_kdtree = nullptr) {
    const auto rtree_start = std::chrono::high_resolution_clock::now();
    const auto rtree_rows = rtree.searchKNN(query, k);
    const auto rtree_end = std::chrono::high_resolution_clock::now();

    const auto kd_start = std::chrono::high_resolution_clock::now();
    const auto kd_rows = kdtree.searchKNN(query, k);
    const auto kd_end = std::chrono::high_resolution_clock::now();

    const auto brute_start = std::chrono::high_resolution_clock::now();
    const auto brute_rows = bruteForceKNN(vectors, query, k);
    const auto brute_end = std::chrono::high_resolution_clock::now();

    const long long rtree_us = std::chrono::duration_cast<std::chrono::microseconds>(rtree_end - rtree_start).count();
    const long long kdtree_us = std::chrono::duration_cast<std::chrono::microseconds>(kd_end - kd_start).count();
    const long long brute_us = std::chrono::duration_cast<std::chrono::microseconds>(brute_end - brute_start).count();

    const std::size_t denom = brute_rows.size();
    const std::size_t rtree_hits = hitsAtK(rtree_rows, brute_rows);
    const std::size_t kdtree_hits = hitsAtK(kd_rows, brute_rows);

    const double rtree_recall = denom == 0 ? 0.0 : static_cast<double>(rtree_hits) / static_cast<double>(denom);
    const double kdtree_recall = denom == 0 ? 0.0 : static_cast<double>(kdtree_hits) / static_cast<double>(denom);

    if (out_rtree != nullptr) {
        *out_rtree = rtree_rows;
    }
    if (out_kdtree != nullptr) {
        *out_kdtree = kd_rows;
    }

    return {
        query_id_label,
        rtree_us,
        kdtree_us,
        brute_us,
        rtree_hits,
        kdtree_hits,
        denom,
        rtree_recall,
        kdtree_recall,
    };
}

AnalysisMetrics runSingleAnalysis(
    const RTreeIndex& rtree,
    const KDTree& kdtree,
    const std::vector<StoredVector>& vectors,
    const std::unordered_map<uint64_t, std::vector<float>>& by_id,
    uint64_t query_id,
    std::size_t k) {
    const auto it = by_id.find(query_id);
    if (it == by_id.end()) {
        throw std::runtime_error("query_id not found in embedding store");
    }
    return runSingleAnalysisRaw(rtree, kdtree, vectors, it->second, query_id, k);
}

void writeCsv(const std::string& path, const std::vector<AnalysisMetrics>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open CSV output path: " + path);
    }

    out << "query_id,rtree_us,kdtree_us,brute_us,rtree_hits,kdtree_hits,denom,rtree_recall,kdtree_recall\n";
    for (const auto& row : rows) {
        out << row.query_id << ","
            << row.rtree_us << ","
            << row.kdtree_us << ","
            << row.brute_us << ","
            << row.rtree_hits << ","
            << row.kdtree_hits << ","
            << row.denom << ","
            << row.rtree_recall << ","
            << row.kdtree_recall << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <db_file> <query_id|all|all:N|vec:f1,f2,...> <k> [csv_output_path]\n";
        return 1;
    }

    const std::string db_file = argv[1];
    const std::string query_selector = argv[2];
    const std::size_t k = static_cast<std::size_t>(std::stoull(argv[3]));
    const bool emit_csv = argc >= 5;
    const std::string csv_path = emit_csv ? argv[4] : std::string();

    if (k == 0) {
        std::cerr << "k must be >= 1\n";
        return 1;
    }

    try {
        uint16_t dims = 0;
        std::vector<StoredVector> vectors;
        {
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

        std::vector<uint64_t> query_ids;
        std::size_t all_limit = 0;
        std::vector<float> parsed_vec;
        const bool is_custom_vec = parseVecSelector(query_selector, parsed_vec);

        if (is_custom_vec) {
            if (parsed_vec.size() != dims) {
                throw std::runtime_error("Custom vector dimension mismatch");
            }
            query_ids.push_back(999999);
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

        StorageManager::disk_reads = 0;
        StorageManager::disk_writes = 0;

        const std::string index_db_file = db_file + ".rtree_tmp.db";
        const bool build_index = !std::filesystem::exists(index_db_file);

        StorageManager index_storage;
        index_storage.open(index_db_file);
        BufferPoolManager bpm(2048, &index_storage);

        RTreeIndex* rtree_ptr = nullptr;
        if (build_index) {
            rtree_ptr = new RTreeIndex(&bpm, dims);
            std::cout << "Building R-Tree index... (this may take a moment)\n";
            for (const auto& vec : unique_vectors) {
                rtree_ptr->insertPoint(vec.values, vec.id);
            }
            bpm.flushAllPages();
            std::cout << "R-Tree index built.\n";
        } else {
            rtree_ptr = new RTreeIndex(&bpm, static_cast<uint32_t>(0));
            std::cout << "Loaded existing R-Tree index from " << index_db_file << "\n";
        }
        RTreeIndex& rtree = *rtree_ptr;

        std::vector<KDEntry> kd_entries;
        kd_entries.reserve(unique_vectors.size());
        for (const auto& vec : unique_vectors) {
            kd_entries.push_back({vec.id, vec.values});
        }

        KDTree kdtree(dims);
        kdtree.build(kd_entries);

        std::vector<std::pair<float, uint64_t>> out_rtree;
        std::vector<std::pair<float, uint64_t>> out_kdtree;

        std::vector<AnalysisMetrics> all_metrics;
        all_metrics.reserve(query_ids.size());

        if (is_custom_vec) {
            all_metrics.push_back(
                runSingleAnalysisRaw(rtree, kdtree, unique_vectors, parsed_vec, 999999, k, &out_rtree, &out_kdtree));
        } else {
            for (uint64_t qid : query_ids) {
                all_metrics.push_back(runSingleAnalysis(rtree, kdtree, unique_vectors, by_id, qid, k));
            }
        }

        long long total_rtree_us = 0;
        long long total_kdtree_us = 0;
        long long total_brute_us = 0;
        std::size_t total_rtree_hits = 0;
        std::size_t total_kdtree_hits = 0;
        std::size_t total_denom = 0;

        for (const auto& row : all_metrics) {
            total_rtree_us += row.rtree_us;
            total_kdtree_us += row.kdtree_us;
            total_brute_us += row.brute_us;
            total_rtree_hits += row.rtree_hits;
            total_kdtree_hits += row.kdtree_hits;
            total_denom += row.denom;
        }

        const double avg_rtree_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_rtree_us) / static_cast<double>(all_metrics.size());
        const double avg_kdtree_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_kdtree_us) / static_cast<double>(all_metrics.size());
        const double avg_brute_us = all_metrics.empty() ? 0.0
            : static_cast<double>(total_brute_us) / static_cast<double>(all_metrics.size());

        const double avg_rtree_recall = total_denom == 0 ? 0.0
            : static_cast<double>(total_rtree_hits) / static_cast<double>(total_denom);
        const double avg_kdtree_recall = total_denom == 0 ? 0.0
            : static_cast<double>(total_kdtree_hits) / static_cast<double>(total_denom);

        std::cout << "KD Analysis Benchmark\n";
        std::cout << "  vectors(raw): " << vectors.size() << "\n";
        std::cout << "  vectors(unique): " << unique_vectors.size() << "\n";
        std::cout << "  dims: " << dims << "\n";
        std::cout << "  query_selector: " << query_selector << "\n";
        std::cout << "  queries_run: " << all_metrics.size() << "\n";
        std::cout << "  k: " << k << "\n";
        std::cout << "\n";

        std::cout << "Average R-tree KNN latency: " << avg_rtree_us << " us\n";
        std::cout << "Average KD-tree KNN latency: " << avg_kdtree_us << " us\n";
        std::cout << "Average brute-force latency: " << avg_brute_us << " us\n";
        std::cout << "Average R-tree recall@k: " << avg_rtree_recall
                  << " (" << total_rtree_hits << "/" << total_denom << ")\n";
        std::cout << "Average KD-tree recall@k: " << avg_kdtree_recall
                  << " (" << total_kdtree_hits << "/" << total_denom << ")\n";

        std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                  << ", writes=" << StorageManager::disk_writes.load() << "\n";
        std::cout << "Buffer Pool hit rate (BPM): " << (bpm.getHitRate() * 100.0)
                  << "% (hits=" << bpm.getHitCount()
                  << ", fetches=" << bpm.getFetchCount()
                  << ", misses=" << bpm.getMissCount() << ")\n";
        std::cout << "RTree metadata page id: " << rtree.getMetaPageId() << "\n";

        const std::size_t print_n = std::min<std::size_t>(5, all_metrics.size());
        std::cout << "\nFirst " << print_n
                  << " rows (query_id, rtree_us, kdtree_us, brute_us, rtree_recall, kdtree_recall):\n";
        for (std::size_t i = 0; i < print_n; ++i) {
            std::cout << "  " << i + 1 << ". "
                      << all_metrics[i].query_id << ", "
                      << all_metrics[i].rtree_us << ", "
                      << all_metrics[i].kdtree_us << ", "
                      << all_metrics[i].brute_us << ", "
                      << all_metrics[i].rtree_recall << ", "
                      << all_metrics[i].kdtree_recall << "\n";
        }

        if (emit_csv) {
            writeCsv(csv_path, all_metrics);
            std::cout << "CSV written: " << csv_path << "\n";
        }

        if (is_custom_vec) {
            std::cout << "\nCustom Vector Neighbors (R-tree):\n";
            for (const auto& row : out_rtree) {
                std::cout << "{ \"id\": " << row.second << ", \"distance\": " << row.first << " }\n";
            }

            std::cout << "\nCustom Vector Neighbors (KD-tree):\n";
            for (const auto& row : out_kdtree) {
                std::cout << "{ \"id\": " << row.second << ", \"distance\": " << row.first << " }\n";
            }
        }

        delete rtree_ptr;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
