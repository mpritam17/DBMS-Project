#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class StorageManager {
public:
    static constexpr std::size_t kPageSize = 4096;

    StorageManager();
    ~StorageManager();

    void open(const std::string& file_path);
    void close();

    std::uint64_t pageCount() const;

    std::vector<std::uint8_t> readPage(std::uint64_t page_id);
    void writePage(std::uint64_t page_id, const std::vector<std::uint8_t>& page_data);

    std::uint64_t allocatePage();
    void flush();

private:
    std::fstream file_;
    std::string file_path_;
    std::uint64_t num_pages_;
    bool is_open_;

    void ensureOpen() const;
};
