#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "storage_manager.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kInvalidPageId = std::numeric_limits<uint32_t>::max();

RTreeNodePage loadNode(BufferPoolManager& buffer_pool_manager, uint32_t page_id) {
    Page* page = buffer_pool_manager.fetchPage(page_id);
    assert(page != nullptr);
    page->RLock();
    std::vector<uint8_t> raw(page->getData(), page->getData() + PageLayout::kPageSize);
    page->RUnlock();
    buffer_pool_manager.unpinPage(page_id, false);
    return RTreeNodePage(raw);
}

void collectLeafValues(BufferPoolManager& buffer_pool_manager, uint32_t page_id, std::vector<uint64_t>& values) {
    RTreeNodePage node = loadNode(buffer_pool_manager, page_id);
    if (node.isLeaf()) {
        for (const RTreeEntry& entry : node.getEntries()) {
            values.push_back(entry.value);
        }
        return;
    }

    for (const RTreeEntry& entry : node.getEntries()) {
        RTreeNodePage child = loadNode(buffer_pool_manager, static_cast<uint32_t>(entry.value));
        assert(child.getParentPageId() == node.getPageId());
        collectLeafValues(buffer_pool_manager, child.getPageId(), values);
    }
}

} // namespace

int main() {
    const std::string db_file = "rtree_insert_test.db";
    std::remove(db_file.c_str());

    StorageManager disk_manager;
    disk_manager.open(db_file);

    {
        BufferPoolManager buffer_pool_manager(16, &disk_manager);
        RTreeIndex index(&buffer_pool_manager, 128);

        for (uint64_t value = 0; value < 10; ++value) {
            std::vector<float> point(128, static_cast<float>(value));
            point[1] = static_cast<float>(value * 2);
            index.insertPoint(point, value);
        }

        assert(index.getHeight() >= 2);

        RTreeNodePage root = loadNode(buffer_pool_manager, index.getRootPageId());
        assert(!root.isLeaf());
        assert(root.getParentPageId() == kInvalidPageId);

        BoundingBox root_mbr = root.computeNodeMBR();
        assert(root_mbr.lower_bounds[0] == 0.0f);
        assert(root_mbr.upper_bounds[0] == 9.0f);
        assert(root_mbr.lower_bounds[1] == 0.0f);
        assert(root_mbr.upper_bounds[1] == 18.0f);

        std::vector<uint64_t> values;
        collectLeafValues(buffer_pool_manager, index.getRootPageId(), values);
        std::sort(values.begin(), values.end());

        assert(values.size() == 10);
        for (uint64_t value = 0; value < 10; ++value) {
            assert(values[value] == value);
        }

        buffer_pool_manager.flushAllPages();
    }

    disk_manager.close();
    std::remove(db_file.c_str());
    std::cout << "R-tree insert and split test passed.\n";
    return 0;
}
