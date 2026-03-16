#include "lru_replacer.h"

LRUReplacer::LRUReplacer(size_t num_pages) : max_size_(num_pages) {}

LRUReplacer::~LRUReplacer() = default;

bool LRUReplacer::victim(uint32_t* frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // First, evict pages that were only accessed once (e.g. table scans)
    if (!fifo_list_.empty()) {
        *frame_id = fifo_list_.back();
        fifo_map_.erase(*frame_id);
        fifo_list_.pop_back();
        access_count_[*frame_id] = 0; // reset history on eviction
        return true;
    }
    
    // If no 1-hit wonders, evict from hot set (LRU)
    if (!lru_list_.empty()) {
        *frame_id = lru_list_.back();
        lru_map_.erase(*frame_id);
        lru_list_.pop_back();
        access_count_[*frame_id] = 0;
        return true;
    }

    return false;
}

void LRUReplacer::pin(uint32_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it_fifo = fifo_map_.find(frame_id);
    if (it_fifo != fifo_map_.end()) {
        fifo_list_.erase(it_fifo->second);
        fifo_map_.erase(it_fifo);
        return;
    }

    auto it_lru = lru_map_.find(frame_id);
    if (it_lru != lru_map_.end()) {
        lru_list_.erase(it_lru->second);
        lru_map_.erase(it_lru);
    }
}

void LRUReplacer::unpin(uint32_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (fifo_map_.find(frame_id) != fifo_map_.end() || 
        lru_map_.find(frame_id) != lru_map_.end()) {
        return; // Already in the replacer
    }

    access_count_[frame_id]++;

    // LRU-K (where k=2) Logic
    if (access_count_[frame_id] == 1) {
        fifo_list_.push_front(frame_id);
        fifo_map_[frame_id] = fifo_list_.begin();
    } else {
        lru_list_.push_front(frame_id);
        lru_map_[frame_id] = lru_list_.begin();
    }
}

size_t LRUReplacer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fifo_map_.size() + lru_map_.size();
}
