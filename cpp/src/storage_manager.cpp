#include "storage_manager.h"

#include <filesystem>
#include <stdexcept>

StorageManager::StorageManager() : num_pages_(0), is_open_(false) {}

StorageManager::~StorageManager() {
    if (is_open_) {
        close();
    }
}

void StorageManager::open(const std::string& file_path) {
    if (is_open_) {
        close();
    }

    file_path_ = file_path;

    if (!std::filesystem::exists(file_path_)) {
        std::ofstream create_file(file_path_, std::ios::binary);
        if (!create_file) {
            throw std::runtime_error("Failed to create database file: " + file_path_);
        }
    }

    file_.open(file_path_, std::ios::in | std::ios::out | std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open database file: " + file_path_);
    }

    const auto file_size = std::filesystem::file_size(file_path_);
    num_pages_ = static_cast<std::uint64_t>(file_size / kPageSize);
    is_open_ = true;
}

void StorageManager::close() {
    ensureOpen();
    file_.flush();
    file_.close();
    is_open_ = false;
}

std::uint64_t StorageManager::pageCount() const {
    return num_pages_;
}

std::vector<std::uint8_t> StorageManager::readPage(std::uint64_t page_id) {
    ensureOpen();

    if (page_id >= num_pages_) {
        throw std::out_of_range("Page id out of bounds");
    }

    std::vector<std::uint8_t> buffer(kPageSize, 0);
    const auto offset = static_cast<std::streamoff>(page_id * kPageSize);

    file_.seekg(offset);
    file_.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(kPageSize));

    if (file_.gcount() != static_cast<std::streamsize>(kPageSize)) {
        throw std::runtime_error("Failed to read full page");
    }

    return buffer;
}

void StorageManager::writePage(std::uint64_t page_id, const std::vector<std::uint8_t>& page_data) {
    ensureOpen();

    if (page_data.size() != kPageSize) {
        throw std::invalid_argument("writePage expects exactly 4096 bytes");
    }

    if (page_id > num_pages_) {
        throw std::out_of_range("Page id cannot skip unallocated pages");
    }

    const auto offset = static_cast<std::streamoff>(page_id * kPageSize);

    file_.seekp(offset);
    file_.write(reinterpret_cast<const char*>(page_data.data()), static_cast<std::streamsize>(kPageSize));

    if (!file_) {
        throw std::runtime_error("Failed to write page");
    }

    if (page_id == num_pages_) {
        ++num_pages_;
    }
}

std::uint64_t StorageManager::allocatePage() {
    ensureOpen();

    const std::uint64_t new_page_id = num_pages_;
    std::vector<std::uint8_t> empty_page(kPageSize, 0);
    writePage(new_page_id, empty_page);
    return new_page_id;
}

void StorageManager::flush() {
    ensureOpen();
    file_.flush();
}

void StorageManager::ensureOpen() const {
    if (!is_open_) {
        throw std::runtime_error("StorageManager file is not open");
    }
}
