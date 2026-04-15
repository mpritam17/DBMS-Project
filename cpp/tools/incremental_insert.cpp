#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "slotted_page.h"
#include "storage_manager.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSlottedPageMagic = 0x50414745;  // "PAGE"
constexpr uint32_t kRTreeMetaMagic = 0x52544958;    // 'RTIX'
constexpr std::size_t kDefaultBpmPages = 2048;

struct StoredVector {
    uint64_t id;
    std::vector<float> values;
};

struct ScanSummary {
    uint16_t dims = 0;
    uint64_t vector_count = 0;
    uint64_t max_id = 0;
    bool has_ids = false;
    bool id_exists = false;
};

void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <db_file> <id|auto> <vec:x,y,...|x,y,...>"
              << " [--index-db <path>] [--bpm-pages <N>]\n";
}

std::vector<float> parseVectorSelector(const std::string& selector) {
    std::string values = selector;
    const std::string prefix = "vec:";
    if (values.rfind(prefix, 0) == 0) {
        values = values.substr(prefix.size());
    }

    if (values.empty()) {
        throw std::runtime_error("Vector selector is empty");
    }

    std::vector<float> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = values.find(',', start);
        const std::string token = comma == std::string::npos
            ? values.substr(start)
            : values.substr(start, comma - start);
        if (token.empty()) {
            throw std::runtime_error("Invalid vector selector: empty coordinate");
        }
        out.push_back(std::stof(token));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if (out.empty()) {
        throw std::runtime_error("Vector selector produced no coordinates");
    }
    return out;
}

ScanSummary scanEmbeddingStore(StorageManager& storage, const std::optional<uint64_t>& target_id) {
    ScanSummary summary{};

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
            if (summary.dims == 0) {
                summary.dims = item_dims;
            }
            if (item_dims != summary.dims) {
                continue;
            }

            uint64_t id = 0;
            std::memcpy(&id, item_data, sizeof(uint64_t));

            ++summary.vector_count;
            if (!summary.has_ids) {
                summary.max_id = id;
                summary.has_ids = true;
            } else if (id > summary.max_id) {
                summary.max_id = id;
            }
            if (target_id.has_value() && id == *target_id) {
                summary.id_exists = true;
            }
        }
    }

    return summary;
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

void appendToEmbeddingStore(StorageManager& storage, uint64_t id, const std::vector<float>& vector) {
    const std::size_t payload_bytes = sizeof(uint64_t) + (vector.size() * sizeof(float));
    std::vector<uint8_t> payload(payload_bytes);
    std::memcpy(payload.data(), &id, sizeof(uint64_t));
    std::memcpy(payload.data() + sizeof(uint64_t), vector.data(), vector.size() * sizeof(float));

    for (uint64_t page_id = 0; page_id < storage.pageCount(); ++page_id) {
        const std::vector<uint8_t> raw = storage.readPage(page_id);
        SlottedPage page(raw);
        const PageHeader* header = page.getHeader();
        if (header->magic != kSlottedPageMagic || header->page_type != 0) {
            continue;
        }

        if (page.addItem(payload.data(), payload.size())) {
            storage.writePage(page_id, page.getRawData());
            return;
        }
    }

    const uint64_t new_page_id = storage.allocatePage();
    SlottedPage page(static_cast<uint32_t>(new_page_id));
    if (!page.addItem(payload.data(), payload.size())) {
        throw std::runtime_error("Vector entry is too large to fit in a new slotted page");
    }
    storage.writePage(new_page_id, page.getRawData());
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

bool readIndexState(const std::string& state_file, uint16_t* dims, uint64_t* count) {
    std::ifstream in(state_file);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    bool found_dims = false;
    bool found_count = false;
    while (std::getline(in, line)) {
        if (line.rfind("dims=", 0) == 0) {
            *dims = static_cast<uint16_t>(std::stoul(line.substr(5)));
            found_dims = true;
            continue;
        }
        if (line.rfind("vectors=", 0) == 0) {
            *count = static_cast<uint64_t>(std::stoull(line.substr(8)));
            found_count = true;
            continue;
        }
    }
    return found_dims && found_count;
}

void writeIndexState(const std::string& state_file, uint16_t dims, uint64_t count) {
    std::ofstream out(state_file, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write index state file: " + state_file);
    }
    out << "dims=" << dims << "\n";
    out << "vectors=" << count << "\n";
}

void rebuildIndexFromStore(
    const std::string& index_db_file,
    StorageManager& embedding_storage,
    uint16_t expected_dims,
    std::size_t bpm_pages) {
    if (std::filesystem::exists(index_db_file)) {
        std::filesystem::remove(index_db_file);
    }

    uint16_t loaded_dims = 0;
    const std::vector<StoredVector> vectors = loadVectorsFromEmbeddingStore(embedding_storage, &loaded_dims);
    if (vectors.empty()) {
        throw std::runtime_error("Cannot rebuild index: embedding store is empty");
    }
    if (loaded_dims != expected_dims) {
        throw std::runtime_error("Cannot rebuild index: dimension mismatch between index target and store");
    }

    StorageManager index_storage;
    index_storage.open(index_db_file);
    BufferPoolManager bpm(bpm_pages, &index_storage);

    RTreeIndex index(&bpm, loaded_dims);
    for (const auto& vec : vectors) {
        index.insertPoint(vec.values, vec.id);
    }
    bpm.flushAllPages();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string db_file = argv[1];
    const std::string id_token = argv[2];
    const std::string vec_selector = argv[3];

    std::string index_db_file;
    std::size_t bpm_pages = kDefaultBpmPages;

    int argi = 4;
    while (argi < argc) {
        const std::string arg = argv[argi];
        if (arg == "--index-db") {
            if (argi + 1 >= argc) {
                throw std::runtime_error("--index-db requires a value");
            }
            index_db_file = argv[argi + 1];
            argi += 2;
            continue;
        }
        if (arg == "--bpm-pages") {
            if (argi + 1 >= argc) {
                throw std::runtime_error("--bpm-pages requires N >= 1");
            }
            const std::size_t parsed = static_cast<std::size_t>(std::stoull(argv[argi + 1]));
            if (parsed == 0) {
                throw std::runtime_error("--bpm-pages requires N >= 1");
            }
            bpm_pages = parsed;
            argi += 2;
            continue;
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (index_db_file.empty()) {
        index_db_file = db_file + ".rtree_tmp.db";
    }
    const std::string index_state_file = index_db_file + ".state";

    try {
        const std::vector<float> vector = parseVectorSelector(vec_selector);
        if (vector.empty()) {
            throw std::runtime_error("Parsed vector is empty");
        }

        const bool auto_id = (id_token == "auto");
        std::optional<uint64_t> requested_id;
        if (!auto_id) {
            requested_id = static_cast<uint64_t>(std::stoull(id_token));
        }

        StorageManager storage;
        storage.open(db_file);

        const ScanSummary summary = scanEmbeddingStore(storage, requested_id);
        const uint16_t dims = summary.dims == 0
            ? static_cast<uint16_t>(vector.size())
            : summary.dims;

        if (dims == 0) {
            throw std::runtime_error("Could not determine vector dimensions");
        }
        if (vector.size() != dims) {
            throw std::runtime_error("Vector dimensions do not match embedding store dimensions");
        }

        uint64_t insert_id = 0;
        if (auto_id) {
            insert_id = summary.has_ids ? (summary.max_id + 1) : 0;
        } else {
            insert_id = *requested_id;
            if (summary.id_exists) {
                throw std::runtime_error("ID already exists in embedding store");
            }
        }

        appendToEmbeddingStore(storage, insert_id, vector);
        storage.flush();

        const uint64_t new_vector_count = summary.vector_count + 1;
        bool index_up_to_date = isReusableTempIndex(index_db_file, dims);

        uint16_t state_dims = 0;
        uint64_t state_count = 0;
        const bool has_state = readIndexState(index_state_file, &state_dims, &state_count);
        if (!has_state || state_dims != dims || state_count != summary.vector_count) {
            index_up_to_date = false;
        }

        if (!index_up_to_date) {
            std::cout << "Index missing/stale. Rebuilding R-tree once before incremental updates...\n";
            rebuildIndexFromStore(index_db_file, storage, dims, bpm_pages);
        } else {
            try {
                StorageManager index_storage;
                index_storage.open(index_db_file);
                BufferPoolManager bpm(bpm_pages, &index_storage);
                RTreeIndex index(&bpm, static_cast<uint32_t>(0));
                index.insertPoint(vector, insert_id);
                bpm.flushAllPages();
            } catch (...) {
                std::cout << "Existing index could not be updated incrementally. Rebuilding...\n";
                rebuildIndexFromStore(index_db_file, storage, dims, bpm_pages);
            }
        }

        writeIndexState(index_state_file, dims, new_vector_count);

        std::cout << "Incremental insert complete\n";
        std::cout << "  db_file: " << db_file << "\n";
        std::cout << "  index_db_file: " << index_db_file << "\n";
        std::cout << "  id: " << insert_id << "\n";
        std::cout << "  dims: " << dims << "\n";
        std::cout << "  vectors_total: " << new_vector_count << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
