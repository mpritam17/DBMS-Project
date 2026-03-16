#include "vec1_reader.h"
#include "slotted_page.h"
#include "storage_manager.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    // 1. Generate fake items: e.g. ID = uint64_t, vector = 128 floats
    size_t vector_dims = 128;
    size_t item_size = sizeof(uint64_t) + vector_dims * sizeof(float); // 8 + 512 = 520 bytes

    std::vector<uint8_t> item1(item_size, 1);
    std::vector<uint8_t> item2(item_size, 2);
    
    // We expect payload size = 4032 bytes.
    // Each item takes 520 bytes + 4 bytes slot = 524 bytes.
    // 4032 / 524 = 7.69 => 7 items should fit exactly.
    size_t max_items = PageLayout::kPayloadSize / (item_size + sizeof(Slot));

    StorageManager storage;
    const std::string test_db = "packing_test.db";
    std::remove(test_db.c_str());
    storage.open(test_db);
    
    uint32_t page_id = storage.allocatePage();
    SlottedPage page(page_id);

    std::cout << "Testing packing for " << item_size << " byte items...\n";
    std::cout << "Expected max items per page: " << max_items << "\n";

    size_t loaded = 0;
    while (page.addItem(item1.data(), item_size)) {
        loaded++;
    }

    std::cout << "Actually packed: " << loaded << "\n";
    if (loaded != max_items) {
        std::cerr << "Mismatch packed items! Packed: " << loaded << ", Expected: " << max_items << "\n";
        return 1;
    }

    storage.writePage(page_id, page.getRawData());
    storage.flush();

    // 2. Reload and Validate integrity
    std::vector<uint8_t> reloaded_raw = storage.readPage(page_id);
    SlottedPage reloaded_page(reloaded_raw);

    const PageHeader* hdr = reloaded_page.getHeader();
    if (hdr->item_count != max_items) {
        std::cerr << "Reloaded page has wrong item count: " << hdr->item_count << "\n";
        return 1;
    }

    for (uint32_t i = 0; i < hdr->item_count; ++i) {
        auto [ptr, len] = reloaded_page.getItem(i);
        if (len != item_size) {
            std::cerr << "Item length mismatch at index " << i << "!\n";
            return 1;
        }
        
        // Items were filled with '1', test validation
        if (ptr[0] != 1) {
            std::cerr << "Item content mismatch at index " << i << "!\n";
            return 1;
        }
    }

    std::cout << "Packing and reload integrity test passed successfully.\n";

    storage.close();
    std::remove(test_db.c_str());
    return 0;
}
