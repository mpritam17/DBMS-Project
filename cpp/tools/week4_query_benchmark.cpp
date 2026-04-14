#include "buffer_pool_manager.h"
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
#include <set>
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

struct QueryMetrics {
    uint64_t query_id;
    long long rtree_us;
    long long brute_us;
    std::size_t hits;
    std::size_t denom;
    double recall;
};

struct PointQueryMetrics {
    uint64_t query_id;
    long long point_us;
    long long brute_us;
    std::size_t rtree_hits;
    std::size_t brute_hits;
    bool exact_match;
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

std::vector<uint64_t> bruteForceExactPoint(
    const std::vector<StoredVector>& vectors,
    const std::vector<float>& query) {
    std::vector<uint64_t> matches;
    for (const auto& vec : vectors) {
        if (vec.values.size() != query.size()) {
            continue;
        }
        if (std::equal(vec.values.begin(), vec.values.end(), query.begin())) {
            matches.push_back(vec.id);
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
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

bool parsePointAllSelector(const std::string& selector, std::size_t* out_limit) {
    if (selector == "pointall") {
        if (out_limit != nullptr) {
            *out_limit = 0;
        }
        return true;
    }

    const std::string prefix = "pointall:";
    if (selector.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string limit_str = selector.substr(prefix.size());
    if (limit_str.empty()) {
        throw std::runtime_error("Invalid query selector 'pointall:'; expected pointall:<positive_integer>");
    }

    const std::size_t limit = static_cast<std::size_t>(std::stoull(limit_str));
    if (limit == 0) {
        throw std::runtime_error("Invalid query selector; pointall:<N> requires N >= 1");
    }

    if (out_limit != nullptr) {
        *out_limit = limit;
    }
    return true;
}

bool parsePointIdSelector(const std::string& selector, uint64_t* out_id) {
    const std::string prefix = "pointid:";
    if (selector.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string id_str = selector.substr(prefix.size());
    if (id_str.empty()) {
        throw std::runtime_error("Invalid pointid selector; expected pointid:<non_negative_integer>");
    }

    const uint64_t id = static_cast<uint64_t>(std::stoull(id_str));
    if (out_id != nullptr) {
        *out_id = id;
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

bool parsePointVecSelector(const std::string& selector, std::vector<float>& out_vec) {
    const std::string prefix = "pointvec:";
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

QueryMetrics benchmarkSingleQueryRaw(
    const RTreeIndex& index,
    const std::vector<StoredVector>& unique_vectors,
    const std::vector<float>& query,
    uint64_t query_id_label,
    std::size_t k,
    std::vector<std::pair<float, uint64_t>>* out_results = nullptr) {
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

    if (out_results) {
        *out_results = rtree_results;
    }

    return {query_id_label, rtree_us, brute_us, hits, denom, recall};
}

QueryMetrics benchmarkSingleQuery(
    const RTreeIndex& index,
    const std::vector<StoredVector>& unique_vectors,
    const std::unordered_map<uint64_t, std::vector<float>>& by_id,
    uint64_t query_id,
    std::size_t k) {
    const auto query_it = by_id.find(query_id);
    if (query_it == by_id.end()) {
        throw std::runtime_error("query_id not found in embedding store");
    }
    return benchmarkSingleQueryRaw(index, unique_vectors, query_it->second, query_id, k);
}

PointQueryMetrics benchmarkSinglePointQueryRaw(
    const RTreeIndex& index,
    const std::vector<StoredVector>& unique_vectors,
    const std::vector<float>& query,
    uint64_t query_id_label,
    std::vector<uint64_t>* out_matches = nullptr) {
    const auto point_start = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> rtree_matches = index.searchExactPoint(query);
    const auto point_end = std::chrono::high_resolution_clock::now();

    const auto brute_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint64_t> brute_matches = bruteForceExactPoint(unique_vectors, query);
    const auto brute_end = std::chrono::high_resolution_clock::now();

    std::sort(rtree_matches.begin(), rtree_matches.end());
    const bool exact_match = (rtree_matches == brute_matches);

    if (out_matches != nullptr) {
        *out_matches = rtree_matches;
    }

    return {
        query_id_label,
        std::chrono::duration_cast<std::chrono::microseconds>(point_end - point_start).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(brute_end - brute_start).count(),
        rtree_matches.size(),
        brute_matches.size(),
        exact_match,
    };
}

PointQueryMetrics benchmarkSinglePointQuery(
    const RTreeIndex& index,
    const std::vector<StoredVector>& unique_vectors,
    const std::unordered_map<uint64_t, std::vector<float>>& by_id,
    uint64_t query_id) {
    const auto query_it = by_id.find(query_id);
    if (query_it == by_id.end()) {
        throw std::runtime_error("point query_id not found in embedding store");
    }
    return benchmarkSinglePointQueryRaw(index, unique_vectors, query_it->second, query_id);
}

void writeCsv(const std::string& csv_path, const std::vector<QueryMetrics>& rows) {
    std::ofstream out(csv_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open CSV output path: " + csv_path);
    }

    out << "query_id,rtree_us,brute_us,hits,denom,recall\n";
    for (const auto& row : rows) {
        out << row.query_id << ","
            << row.rtree_us << ","
            << row.brute_us << ","
            << row.hits << ","
            << row.denom << ","
            << row.recall << "\n";
    }
}

void writePointCsv(const std::string& csv_path, const std::vector<PointQueryMetrics>& rows) {
    std::ofstream out(csv_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open CSV output path: " + csv_path);
    }

    out << "query_id,point_us,brute_us,rtree_hits,brute_hits,exact_match\n";
    for (const auto& row : rows) {
        out << row.query_id << ","
            << row.point_us << ","
            << row.brute_us << ","
            << row.rtree_hits << ","
            << row.brute_hits << ","
            << (row.exact_match ? 1 : 0) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <db_file> <query_selector> <k> [csv_output_path]\n"
                  << "  KNN selectors: <id> | all | all:N | vec:f1,f2,...\n"
                  << "  Exact-point selectors: pointid:<id> | pointall | pointall:N | pointvec:f1,f2,...\n";
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

        std::vector<uint64_t> knn_query_ids;
        std::vector<uint64_t> point_query_ids;

        std::size_t all_limit = 0;
        std::size_t point_all_limit = 0;
        uint64_t point_query_id = 0;

        std::vector<float> parsed_vec_query;
        std::vector<float> parsed_point_vec;

        bool is_custom_vec = parseVecSelector(query_selector, parsed_vec_query);
        const bool is_point_vec = parsePointVecSelector(query_selector, parsed_point_vec);
        const bool is_point_id = parsePointIdSelector(query_selector, &point_query_id);
        const bool is_point_all = parsePointAllSelector(query_selector, &point_all_limit);
        const bool is_point_mode = is_point_vec || is_point_id || is_point_all;

        if (is_custom_vec && is_point_mode) {
            throw std::runtime_error("query selector is ambiguous between KNN and exact-point modes");
        }

        if (is_custom_vec) {
            if (parsed_vec_query.size() != dims) {
                throw std::runtime_error("Custom vector dimension mismatch");
            }
            knn_query_ids.push_back(999999);  // dummy label for output
        } else if (!is_point_mode && parseAllSelector(query_selector, &all_limit)) {
            knn_query_ids.reserve(unique_vectors.size());
            for (const auto& v : unique_vectors) {
                knn_query_ids.push_back(v.id);
            }
            if (all_limit > 0 && knn_query_ids.size() > all_limit) {
                knn_query_ids.resize(all_limit);
            }
        } else if (!is_point_mode) {
            knn_query_ids.push_back(std::stoull(query_selector));
        }

        if (is_point_mode) {
            if (is_point_vec) {
                if (parsed_point_vec.size() != dims) {
                    throw std::runtime_error("Custom point vector dimension mismatch");
                }
            } else if (is_point_all) {
                point_query_ids.reserve(unique_vectors.size());
                for (const auto& v : unique_vectors) {
                    point_query_ids.push_back(v.id);
                }
                if (point_all_limit > 0 && point_query_ids.size() > point_all_limit) {
                    point_query_ids.resize(point_all_limit);
                }
            } else {
                point_query_ids.push_back(point_query_id);
            }
        }

        StorageManager::disk_reads = 0;
        StorageManager::disk_writes = 0;

        // Build the R-tree in a separate temp DB to avoid mixing index pages
        // with embedding slotted pages in the source file. Re-use if exists!
        const std::string index_db_file = db_file + ".rtree_tmp.db";
        bool build_index = !std::filesystem::exists(index_db_file);
        
        StorageManager index_storage;
        index_storage.open(index_db_file);

        // Increased buffer pool size for building the index over 60k vectors smoothly
        BufferPoolManager bpm(2048, &index_storage);
        
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

        if (is_point_mode) {
            std::vector<uint64_t> out_point_matches;
            std::vector<PointQueryMetrics> point_metrics;
            point_metrics.reserve(is_point_vec ? 1 : point_query_ids.size());

            if (is_point_vec) {
                point_metrics.push_back(
                    benchmarkSinglePointQueryRaw(index, unique_vectors, parsed_point_vec, 999999, &out_point_matches));
            } else {
                for (uint64_t qid : point_query_ids) {
                    point_metrics.push_back(benchmarkSinglePointQuery(index, unique_vectors, by_id, qid));
                }
            }

            long long total_point_us = 0;
            long long total_brute_us = 0;
            std::size_t exact_matches = 0;
            for (const auto& m : point_metrics) {
                total_point_us += m.point_us;
                total_brute_us += m.brute_us;
                if (m.exact_match) {
                    ++exact_matches;
                }
            }

            const double avg_point_us = point_metrics.empty()
                ? 0.0
                : static_cast<double>(total_point_us) / static_cast<double>(point_metrics.size());
            const double avg_brute_us = point_metrics.empty()
                ? 0.0
                : static_cast<double>(total_brute_us) / static_cast<double>(point_metrics.size());
            const double match_rate = point_metrics.empty()
                ? 0.0
                : static_cast<double>(exact_matches) / static_cast<double>(point_metrics.size());

            std::cout << "Week 4 Exact-Point Benchmark\n";
            std::cout << "  vectors(raw): " << vectors.size() << "\n";
            std::cout << "  vectors(unique): " << unique_vectors.size() << "\n";
            std::cout << "  dims: " << dims << "\n";
            std::cout << "  query_selector: " << query_selector << "\n";
            std::cout << "  queries_run: " << point_metrics.size() << "\n";
            std::cout << "\n";
            std::cout << "Average exact-point latency: " << avg_point_us << " us\n";
            std::cout << "Average brute-force exact-point latency: " << avg_brute_us << " us\n";
            std::cout << "Exact-point match rate: " << match_rate << " (" << exact_matches
                      << "/" << point_metrics.size() << ")\n";
            std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                      << ", writes=" << StorageManager::disk_writes.load() << "\n";
            const double bpm_hit_rate = bpm.getHitRate();
            std::cout << "Buffer Pool hit rate (BPM): " << (bpm_hit_rate * 100.0)
                      << "% (hits=" << bpm.getHitCount()
                      << ", fetches=" << bpm.getFetchCount()
                      << ", misses=" << bpm.getMissCount() << ")\n";
            std::cout << "RTree metadata page id: " << index.getMetaPageId() << "\n";

            const std::size_t print_n = std::min<std::size_t>(5, point_metrics.size());
            std::cout << "\nFirst " << print_n
                      << " exact-point rows (query_id, point_us, brute_us, rtree_hits, brute_hits, exact_match):\n";
            for (std::size_t i = 0; i < print_n; ++i) {
                std::cout << "  " << i + 1 << ". "
                          << point_metrics[i].query_id << ", "
                          << point_metrics[i].point_us << ", "
                          << point_metrics[i].brute_us << ", "
                          << point_metrics[i].rtree_hits << ", "
                          << point_metrics[i].brute_hits << ", "
                          << (point_metrics[i].exact_match ? 1 : 0) << "\n";
            }

            if (emit_csv) {
                writePointCsv(csv_path, point_metrics);
                std::cout << "CSV written: " << csv_path << "\n";
            }

            if (is_point_vec) {
                std::cout << "\nExact Point Matches:\n";
                if (out_point_matches.empty()) {
                    std::cout << "{ \"id\": null }\n";
                } else {
                    for (uint64_t id : out_point_matches) {
                        std::cout << "{ \"id\": " << id << " }\n";
                    }
                }
            }
        } else {
            std::vector<std::pair<float, uint64_t>> out_results;
            std::vector<QueryMetrics> all_metrics;
            all_metrics.reserve(knn_query_ids.size());

            if (is_custom_vec) {
                all_metrics.push_back(benchmarkSingleQueryRaw(index, unique_vectors, parsed_vec_query, 999999, k, &out_results));
            } else {
                for (uint64_t qid : knn_query_ids) {
                    all_metrics.push_back(benchmarkSingleQuery(index, unique_vectors, by_id, qid, k));
                }
            }

            long long total_rtree_us = 0;
            long long total_brute_us = 0;
            std::size_t total_hits = 0;
            std::size_t total_denom = 0;
            for (const auto& m : all_metrics) {
                total_rtree_us += m.rtree_us;
                total_brute_us += m.brute_us;
                total_hits += m.hits;
                total_denom += m.denom;
            }

            const double avg_rtree_us = all_metrics.empty() ? 0.0
                : static_cast<double>(total_rtree_us) / static_cast<double>(all_metrics.size());
            const double avg_brute_us = all_metrics.empty() ? 0.0
                : static_cast<double>(total_brute_us) / static_cast<double>(all_metrics.size());
            const double avg_recall = total_denom == 0 ? 0.0
                : static_cast<double>(total_hits) / static_cast<double>(total_denom);

            std::cout << "Week 4 Query Benchmark\n";
            std::cout << "  vectors(raw): " << vectors.size() << "\n";
            std::cout << "  vectors(unique): " << unique_vectors.size() << "\n";
            std::cout << "  dims: " << dims << "\n";
            std::cout << "  query_selector: " << query_selector << "\n";
            std::cout << "  queries_run: " << all_metrics.size() << "\n";
            std::cout << "  k: " << k << "\n";
            std::cout << "\n";
            std::cout << "Average R-tree KNN latency: " << avg_rtree_us << " us\n";
            std::cout << "Average brute-force latency: " << avg_brute_us << " us\n";
            std::cout << "Average recall@k: " << avg_recall << " (" << total_hits << "/" << total_denom << ")\n";
            std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                      << ", writes=" << StorageManager::disk_writes.load() << "\n";
            const double bpm_hit_rate = bpm.getHitRate();
            std::cout << "Buffer Pool hit rate (BPM): " << (bpm_hit_rate * 100.0)
                      << "% (hits=" << bpm.getHitCount()
                      << ", fetches=" << bpm.getFetchCount()
                      << ", misses=" << bpm.getMissCount() << ")\n";
            std::cout << "RTree metadata page id: " << index.getMetaPageId() << "\n";

            const std::size_t print_n = std::min<std::size_t>(5, all_metrics.size());
            std::cout << "\nFirst " << print_n << " per-query rows (query_id, rtree_us, brute_us, recall):\n";
            for (std::size_t i = 0; i < print_n; ++i) {
                std::cout << "  " << i + 1 << ". "
                          << all_metrics[i].query_id << ", "
                          << all_metrics[i].rtree_us << ", "
                          << all_metrics[i].brute_us << ", "
                          << all_metrics[i].recall << "\n";
            }

            if (emit_csv) {
                writeCsv(csv_path, all_metrics);
                std::cout << "CSV written: " << csv_path << "\n";
            }

            if (is_custom_vec) {
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
