#pragma once

#include <list>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <optional>

class LRUReplacer {
public:
    explicit LRUReplacer(size_t num_pages);
    ~LRUReplacer();

    // Remove the victim frame as defined by the replacement policy.
    bool victim(uint32_t* frame_id);

    // Pins a frame, indicating that it should not be victimized until it is unpinned.
    void pin(uint32_t frame_id);

    // Unpins a frame, indicating that it can now be victimized.
    void unpin(uint32_t frame_id);

    // Returns the number of frames that are currently in the replacer.
    size_t size() const;

private:
    mutable std::mutex mutex_;
    
    std::list<uint32_t> fifo_list_;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> fifo_map_;

    std::list<uint32_t> lru_list_;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lru_map_;

    std::unordered_map<uint32_t, size_t> access_count_;

    size_t max_size_;
};
