#include "buffer_pool_manager.h"
#include <cassert>

// Buffer Pool Manager Implementation
// Key optimizations:
// 1. Zero-copy I/O: Uses direct pointer-based I/O (readPageTo/writePageFrom)
//    instead of vector copies to eliminate 4KB allocations on every write
// 2. Fine-grained locking: Global latch protects metadata only; expensive
//    disk I/O happens outside global latch with per-page locks
// 3. LRU-2 eviction: Two-queue design prevents sequential scans from evicting hot pages

BufferPoolManager::BufferPoolManager(size_t pool_size, StorageManager* storage_manager)
    : pool_size_(pool_size), disk_manager_(storage_manager) {
    pages_ = std::make_unique<Page[]>(pool_size_);
    replacer_ = std::make_unique<LRUReplacer>(pool_size);
    for (size_t i = 0; i < pool_size_; i++) {
        free_list_.push_back(static_cast<uint32_t>(i));
    }
}

BufferPoolManager::~BufferPoolManager() {
    flushAllPages();
}

bool BufferPoolManager::findVictim(uint32_t* frame_id) {
    if (!free_list_.empty()) {
        *frame_id = free_list_.front();
        free_list_.pop_front();
        return true;
    }
    return replacer_->victim(frame_id);
}

Page* BufferPoolManager::fetchPage(uint32_t page_id) {
    latch_.lock();
    fetch_requests_.fetch_add(1, std::memory_order_relaxed);
    auto it = page_table_.find(page_id);
    if (it != page_table_.end()) {
        fetch_hits_.fetch_add(1, std::memory_order_relaxed);
        uint32_t frame_id = it->second;
        Page* page = &pages_[frame_id];
        page->pin_count_++;
        replacer_->pin(frame_id);
        latch_.unlock();
        return page;
    }
    fetch_misses_.fetch_add(1, std::memory_order_relaxed);
    
    uint32_t frame_id;
    if (!findVictim(&frame_id)) {
        latch_.unlock();
        return nullptr;
    }
    
    Page* page = &pages_[frame_id];
    
    // Unbind the frame from its old page immediately
    if (page->page_id_ != INVALID_PAGE_ID) {
        page_table_.erase(page->page_id_);
    }
    page_table_[page_id] = frame_id;
    page->pin_count_ = 1;

    uint32_t old_page_id = page->page_id_;
    bool is_dirty = page->is_dirty_;
    page->page_id_ = page_id;
    replacer_->pin(frame_id);
    
    latch_.unlock();

    // Now do the expensive I/O OUTSIDE the global latch!
    page->WLock();
    if (is_dirty && old_page_id != INVALID_PAGE_ID) {
        disk_manager_->writePageFrom(old_page_id, page->getData());
    }

    disk_manager_->readPageTo(page_id, page->getData());
    page->is_dirty_ = false;
    page->WUnlock();
    
    return page;
}

bool BufferPoolManager::unpinPage(uint32_t page_id, bool is_dirty) {
    std::lock_guard<std::mutex> lock(latch_);
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) return false;
    
    uint32_t frame_id = it->second;
    Page* page = &pages_[frame_id];
    if (page->pin_count_ <= 0) return false;
    
    page->pin_count_--;
    if (is_dirty) page->is_dirty_ = true;
    if (page->pin_count_ == 0) replacer_->unpin(frame_id);
    return true;
}

bool BufferPoolManager::flushPage(uint32_t page_id) {
    latch_.lock();
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) {
        latch_.unlock();
        return false;
    }
    uint32_t frame_id = it->second;
    Page* page = &pages_[frame_id];
    latch_.unlock();

    page->WLock();
    disk_manager_->writePageFrom(page->page_id_, page->getData());
    page->is_dirty_ = false;
    page->WUnlock();
    return true;
}

Page* BufferPoolManager::newPage(uint32_t* page_id) {
    latch_.lock();
    uint32_t frame_id;
    if (!findVictim(&frame_id)) {
        latch_.unlock();
        return nullptr;
    }
    Page* page = &pages_[frame_id];
    if (page->page_id_ != INVALID_PAGE_ID) {
        page_table_.erase(page->page_id_);
    }
    *page_id = static_cast<uint32_t>(disk_manager_->allocatePage());
    
    uint32_t old_page_id = page->page_id_;
    bool is_dirty = page->is_dirty_;

    page->page_id_ = *page_id;
    page->pin_count_ = 1;
    page_table_[*page_id] = frame_id;
    replacer_->pin(frame_id);
    latch_.unlock();

    page->WLock();
    if (is_dirty && old_page_id != INVALID_PAGE_ID) {
        disk_manager_->writePageFrom(old_page_id, page->getData());
    }
    page->is_dirty_ = true;
    page->resetMemory();
    page->WUnlock();
    
    return page;
}

void BufferPoolManager::flushAllPages() {
    latch_.lock();
    std::vector<Page*> flush_queue;
    flush_queue.reserve(pool_size_);  // Avoid reallocation
    for (size_t i = 0; i < pool_size_; i++) {
        if (pages_[i].page_id_ != INVALID_PAGE_ID && pages_[i].is_dirty_) {
            flush_queue.push_back(&pages_[i]);
        }
    }
    latch_.unlock();

    for (Page* page : flush_queue) {
        page->WLock();
        disk_manager_->writePageFrom(page->page_id_, page->getData());
        page->is_dirty_ = false;
        page->WUnlock();
    }
    disk_manager_->flush();
}

void BufferPoolManager::resetStats() {
    fetch_requests_.store(0, std::memory_order_relaxed);
    fetch_hits_.store(0, std::memory_order_relaxed);
    fetch_misses_.store(0, std::memory_order_relaxed);
}

uint64_t BufferPoolManager::getFetchRequests() const {
    return fetch_requests_.load(std::memory_order_relaxed);
}

uint64_t BufferPoolManager::getFetchHits() const {
    return fetch_hits_.load(std::memory_order_relaxed);
}

uint64_t BufferPoolManager::getFetchMisses() const {
    return fetch_misses_.load(std::memory_order_relaxed);
}

double BufferPoolManager::getHitRate() const {
    const uint64_t requests = getFetchRequests();
    if (requests == 0) {
        return 0.0;
    }
    const uint64_t hits = getFetchHits();
    return static_cast<double>(hits) / static_cast<double>(requests);
}
