#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>

struct Vec1Header {
    char magic[4];
    uint32_t version;
    uint64_t count;
    uint32_t dims;
    uint32_t reserved;
};

struct Vec1Entry {
    uint64_t id;
    std::vector<float> vector;
};

class Vec1Reader {
public:
    explicit Vec1Reader(const std::string& file_path);
    ~Vec1Reader();

    const Vec1Header& getHeader() const { return header_; }
    
    // Reads the next entry. Returns true if successful, false if EOF.
    bool readNext(Vec1Entry& entry);
    
    // Reads all remaining entries into the provided vector.
    void readAll(std::vector<Vec1Entry>& entries);

private:
    std::ifstream file_;
    Vec1Header header_;
    uint64_t current_index_;

    void readHeader();
};
