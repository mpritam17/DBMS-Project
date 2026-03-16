#include "slotted_page.h"
#include <cstring>
#include <stdexcept>

const uint32_t PAGE_MAGIC = 0x50414745; // "PAGE"

SlottedPage::SlottedPage(uint32_t page_id) {
    data_.resize(PageLayout::kPageSize, 0);
    PageHeader* header = getHeaderMut();
    header->magic = PAGE_MAGIC;
    header->page_type = 0; // standard slotted page
    header->flags = 0;
    header->page_id = page_id;
    header->item_count = 0;
    // Free space starts at the end of the page
    header->free_space_offset = PageLayout::kPageSize;
    // Total free space available is everything after the header
    header->free_space_bytes = PageLayout::kPayloadSize;
}

SlottedPage::SlottedPage(const std::vector<uint8_t>& raw_data) : data_(raw_data) {
    if (data_.size() != PageLayout::kPageSize) {
        throw std::runtime_error("Invalid page size");
    }
}

PageHeader* SlottedPage::getHeaderMut() {
    return reinterpret_cast<PageHeader*>(data_.data());
}

const PageHeader* SlottedPage::getHeader() const {
    return reinterpret_cast<const PageHeader*>(data_.data());
}

Slot* SlottedPage::getSlotArrayMut() {
    return reinterpret_cast<Slot*>(data_.data() + PageLayout::kHeaderSize);
}

const Slot* SlottedPage::getSlotArray() const {
    return reinterpret_cast<const Slot*>(data_.data() + PageLayout::kHeaderSize);
}

bool SlottedPage::addItem(const uint8_t* item_data, size_t item_length) {
    PageHeader* header = getHeaderMut();
    
    // Calculate space needed: the item itself plus a new Slot tracking it
    size_t space_needed = item_length + sizeof(Slot);
    
    if (header->free_space_bytes < space_needed) {
        return false; // Not enough space
    }

    // New offset for the data
    uint16_t new_offset = header->free_space_offset - static_cast<uint16_t>(item_length);
    
    // Copy the data
    std::memcpy(data_.data() + new_offset, item_data, item_length);
    
    // Write the slot
    Slot* slots = getSlotArrayMut();
    slots[header->item_count].offset = new_offset;
    slots[header->item_count].length = static_cast<uint16_t>(item_length);

    // Update header
    header->free_space_offset = new_offset;
    header->item_count++;
    header->free_space_bytes -= static_cast<uint16_t>(space_needed);

    return true;
}

std::pair<const uint8_t*, size_t> SlottedPage::getItem(uint32_t item_index) const {
    const PageHeader* header = getHeader();
    if (item_index >= header->item_count) {
        throw std::out_of_range("Item index out of bounds");
    }
    
    const Slot* slots = getSlotArray();
    const Slot& slot = slots[item_index];
    
    return {data_.data() + slot.offset, slot.length};
}
