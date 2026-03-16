#include "vec1_reader.h"
#include <cstring>
#include <iostream>

Vec1Reader::Vec1Reader(const std::string& file_path) : current_index_(0) {
    file_.open(file_path, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open VEC1 file: " + file_path);
    }
    readHeader();
}

Vec1Reader::~Vec1Reader() {
    if (file_.is_open()) {
        file_.close();
    }
}

void Vec1Reader::readHeader() {
    file_.read(reinterpret_cast<char*>(&header_), sizeof(Vec1Header));
    if (!file_) {
        throw std::runtime_error("Failed to read VEC1 header.");
    }
    
    // Validate magic
    if (std::strncmp(header_.magic, "VEC1", 4) != 0) {
        throw std::runtime_error("Invalid VEC1 file magic.");
    }
}

bool Vec1Reader::readNext(Vec1Entry& entry) {
    if (current_index_ >= header_.count) {
        return false;
    }
    
    file_.read(reinterpret_cast<char*>(&entry.id), sizeof(uint64_t));
    if (!file_) {
        return false;
    }

    entry.vector.resize(header_.dims);
    file_.read(reinterpret_cast<char*>(entry.vector.data()), header_.dims * sizeof(float));
    if (!file_) {
        return false;
    }

    current_index_++;
    return true;
}

void Vec1Reader::readAll(std::vector<Vec1Entry>& entries) {
    entries.reserve(header_.count);
    Vec1Entry entry;
    while (readNext(entry)) {
        entries.push_back(entry);
    }
}
