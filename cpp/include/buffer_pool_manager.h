#pragma once

#include "page.h"
#include "lru_replacer.h"
#include "storage_manager.h"

#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <memory>
#include <atomic>

class BufferPoolManager {
public:
    BufferPoolManager(size_t pool_size, StorageManager* storage_manager);
    ~BufferPoolManager();

    // Fetch the requested page from the buffer pool.
    Page* fetchPage(uint32_t page_id);

    // Unpin the target page from the buffer pool.
    bool unpinPage(uint32_t page_id, bool is_dirty);

    // Flush the target page to disk.
    bool flushPage(uint32_t page_id);

    // Creates a new page in the buffer pool.
    Page* newPage(uint32_t* page_id);

    // Flushes all the pages in the buffer pool to disk.
    void flushAllPages();

    // Buffer-pool fetch statistics used by benchmarks.
    void resetStats();
    uint64_t getFetchRequests() const;
    uint64_t getFetchHits() const;
    uint64_t getFetchMisses() const;
    double getHitRate() const;

protected:
    // Find a free frame. Try free list first, then replacer.
    inline bool findVictim(uint32_t* frame_id);

private:
    size_t pool_size_;
    std::unique_ptr<Page[]> pages_;
    StorageManager* disk_manager_;
    std::unique_ptr<LRUReplacer> replacer_;
    
    // Maps page_id to frame_id
    std::unordered_map<uint32_t, uint32_t> page_table_;
    
    // List of free frames that don't have any pages on them
    std::list<uint32_t> free_list_;

    std::atomic<uint64_t> fetch_requests_{0};
    std::atomic<uint64_t> fetch_hits_{0};
    std::atomic<uint64_t> fetch_misses_{0};
    
    std::mutex latch_;
};
