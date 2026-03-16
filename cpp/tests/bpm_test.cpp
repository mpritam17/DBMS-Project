#include "buffer_pool_manager.h"
#include "storage_manager.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdio>

int main() {
    std::cout << "Starting Buffer Pool Manager Test...\n";
    
    const std::string db_file = "bpm_test.db";
    std::remove(db_file.c_str());

    StorageManager disk_manager;
    disk_manager.open(db_file);

    {
        // Create a very small buffer pool with only 3 frames to force evictions
        BufferPoolManager bpm(3, &disk_manager);

    uint32_t page_id_0;
    Page* page0 = bpm.newPage(&page_id_0);
    assert(page0 != nullptr);
    std::snprintf(reinterpret_cast<char*>(page0->getData()), PageLayout::kPageSize, "Hello Page 0");
    bpm.unpinPage(page_id_0, true);

    uint32_t page_id_1;
    Page* page1 = bpm.newPage(&page_id_1);
    assert(page1 != nullptr);
    std::snprintf(reinterpret_cast<char*>(page1->getData()), PageLayout::kPageSize, "Hello Page 1");
    bpm.unpinPage(page_id_1, true);

    uint32_t page_id_2;
    Page* page2 = bpm.newPage(&page_id_2);
    assert(page2 != nullptr);
    std::snprintf(reinterpret_cast<char*>(page2->getData()), PageLayout::kPageSize, "Hello Page 2");
    // Leave page 2 pinned for a moment

    // We have 3 frames: page0(unpinned), page1(unpinned), page2(pinned)
    // Allocating a new page should evict the LRU, which is page0
    uint32_t page_id_3;
    Page* page3 = bpm.newPage(&page_id_3);
    assert(page3 != nullptr);
    std::snprintf(reinterpret_cast<char*>(page3->getData()), PageLayout::kPageSize, "Hello Page 3");
    bpm.unpinPage(page_id_3, true);

    // Fetch page 0 again -> should evict page 1 (since page 2 is still pinned)
    page0 = bpm.fetchPage(page_id_0);
    assert(page0 != nullptr);
    assert(std::strcmp(reinterpret_cast<char*>(page0->getData()), "Hello Page 0") == 0);
    bpm.unpinPage(page_id_0, false);
    
    // Now unpin 2 and flush all
    bpm.unpinPage(page_id_2, true);
    bpm.flushAllPages();

    // Destroy bpm before closing disk_manager
    }
    
    disk_manager.close();
    
    std::cout << "Buffer Pool Manager Test Passed!\n";
    std::remove(db_file.c_str());
    return 0;
}
