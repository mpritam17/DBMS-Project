#pragma once

#include "storage_manager.h"
#include <vector>
#include <cstdint>
#include <tuple>

// Slot definition
struct Slot {
    uint16_t offset;
    uint16_t length;
};
static_assert(sizeof(Slot) == 4, "Slot must be 4 bytes");

class SlottedPage {
public:
    // Create a new empty page
    explicit SlottedPage(uint32_t page_id);
    
    // wrap an existing raw page from disk
    explicit SlottedPage(const std::vector<uint8_t>& raw_data);

    // Get the underlying raw data
    const std::vector<uint8_t>& getRawData() const { return data_; }

    // Adds an item to the page. Returns true if successful, false if not enough space.
    bool addItem(const uint8_t* item_data, size_t item_length);

    // Retrieves an item. Returns a pointer to the item and its length.
    std::pair<const uint8_t*, size_t> getItem(uint32_t item_index) const;

    // View the header
    const PageHeader* getHeader() const;

private:
    std::vector<uint8_t> data_;
    
    PageHeader* getHeaderMut();
    Slot* getSlotArrayMut();
    const Slot* getSlotArray() const;
};
