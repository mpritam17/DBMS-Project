#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

// Initial physical page layout constants for Week 1 design.
struct PageLayout {
    static constexpr std::size_t kPageSize = 4096;
    static constexpr std::size_t kHeaderSize = 64;
    static constexpr std::size_t kPayloadSize = kPageSize - kHeaderSize;
};

struct PageHeader {
    std::uint32_t magic;
    std::uint16_t page_type;
    std::uint16_t flags;
    std::uint32_t page_id;
    std::uint32_t item_count;
    std::uint16_t free_space_offset;
    std::uint16_t free_space_bytes;
    std::uint8_t reserved[44];
};

static_assert(sizeof(PageHeader) == PageLayout::kHeaderSize, "PageHeader must be 64 bytes");

class StorageManager {
public:
    static constexpr std::size_t kPageSize = PageLayout::kPageSize;

    StorageManager();
    ~StorageManager();

    void open(const std::string& file_path);
    void close();

    std::uint64_t pageCount() const;

    std::vector<std::uint8_t> readPage(std::uint64_t page_id);
    void writePage(std::uint64_t page_id, const std::vector<std::uint8_t>& page_data);

    // Zero-copy I/O interface for buffer pool manager
    void readPageTo(std::uint64_t page_id, uint8_t* buffer);
    void writePageFrom(std::uint64_t page_id, const uint8_t* buffer);

    std::uint64_t allocatePage();
    void flush();
    
    // Performance metrics
    static std::atomic<int> disk_reads;
    static std::atomic<int> disk_writes;

private:
    std::fstream file_;
    std::string file_path_;
    std::uint64_t num_pages_;
    bool is_open_;
    mutable std::mutex disk_latch_;

    void ensureOpen() const;
};
