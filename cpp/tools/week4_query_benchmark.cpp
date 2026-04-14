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

struct BenchmarkOptions {
    bool fair_mode = false;
    std::size_t bpm_pages = 2048;
    bool bpm_pages_explicit = false;
};

constexpr std::size_t kDefaultBpmPages = 2048;
constexpr std::size_t kAutoBpmMinPages = 2048;
constexpr std::size_t kAutoBpmMaxPages = 8192;
constexpr std::size_t kAutoBpmHeadroomPages = 128;

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <db_file> <query_id|all|all:N|vec:x,y,...> <k>"
                  << " [csv_output_path] [--fair] [--bpm-pages N|auto]\n";
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
        bool build_index = !std::filesystem::exists(index_db_file);
        
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
                (void) index.searchKNN(parsed_vec_query, k);
            } else {
                for (uint64_t qid : query_ids) {
                    const auto query_it = by_id.find(qid);
                    if (query_it != by_id.end()) {
                        (void) index.searchKNN(query_it->second, k);
                    }
                }
            }
        }

        StorageManager::disk_reads = 0;
        StorageManager::disk_writes = 0;

        std::vector<std::pair<float, uint64_t>> out_results;
        std::vector<QueryMetrics> all_metrics;
        all_metrics.reserve(query_ids.size());

        if (is_custom_vec) {
            all_metrics.push_back(benchmarkSingleQueryRaw(index, unique_vectors, parsed_vec_query, 999999, k, &out_results));
        } else {
            for (uint64_t qid : query_ids) {
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
        std::cout << "  fair_mode: " << (options.fair_mode ? "on" : "off") << "\n";
        std::cout << "  bpm_pages: " << options.bpm_pages << "\n";
        std::cout << "\n";
        std::cout << "Average R-tree KNN latency: " << avg_rtree_us << " us\n";
        std::cout << "Average brute-force latency: " << avg_brute_us << " us\n";
        std::cout << "Average recall@k: " << avg_recall << " (" << total_hits << "/" << total_denom << ")\n";
        std::cout << "Disk I/O counters (StorageManager): reads=" << StorageManager::disk_reads.load()
                  << ", writes=" << StorageManager::disk_writes.load() << "\n";
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
