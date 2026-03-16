#pragma once

#include "storage_manager.h"
#include <vector>
#include <cstdint>
#include <shared_mutex>

// A constant representing an invalid page ID
constexpr uint32_t INVALID_PAGE_ID = 0xFFFFFFFF;

// In-memory wrapper for database pages managed by the Buffer Pool Manager.
class Page {
public:
    Page() {
        data_.resize(PageLayout::kPageSize, 0);
        page_id_ = INVALID_PAGE_ID;
        is_dirty_ = false;
        pin_count_ = 0;
    }

    ~Page() = default;

    inline uint8_t* getData() { return data_.data(); }
    inline const uint8_t* getData() const { return data_.data(); }

    inline uint32_t getPageId() const { return page_id_; }
    inline int getPinCount() const { return pin_count_; }
    inline bool isDirty() const { return is_dirty_; }

    // Latches for fine-grained concurrent access
    inline void RLock() { rwlatch_.lock_shared(); }
    inline void RUnlock() { rwlatch_.unlock_shared(); }
    inline void WLock() { rwlatch_.lock(); }
    inline void WUnlock() { rwlatch_.unlock(); }

private:
    friend class BufferPoolManager;

    // Reset memory space dynamically 
    inline void resetMemory() {
        std::fill(data_.begin(), data_.end(), 0);
    }

    std::vector<uint8_t> data_;
    uint32_t page_id_;
    int pin_count_;
    bool is_dirty_;
    std::shared_mutex rwlatch_;
};
