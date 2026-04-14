#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "storage_manager.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    const std::string db_file = "rtree_point_test.db";
    std::remove(db_file.c_str());

    StorageManager disk_manager;
    disk_manager.open(db_file);

    {
        BufferPoolManager buffer_pool_manager(64, &disk_manager);
        RTreeIndex index(&buffer_pool_manager, static_cast<uint16_t>(2));

        index.insertPoint({1.0f, 2.0f}, 10);
        index.insertPoint({1.0f, 2.0f}, 20);
        index.insertPoint({3.0f, 4.0f}, 30);

        auto matches = index.searchExactPoint({1.0f, 2.0f});
        std::sort(matches.begin(), matches.end());
        assert(matches.size() == 2);
        assert(matches[0] == 10);
        assert(matches[1] == 20);

        auto none = index.searchExactPoint({9.0f, 9.0f});
        assert(none.empty());

        bool threw = false;
        try {
            (void)index.searchExactPoint({1.0f});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw && "searchExactPoint must reject dimension mismatch");

        buffer_pool_manager.flushAllPages();
    }

    disk_manager.close();
    std::remove(db_file.c_str());
    std::printf("R-tree exact-point test passed.\n");
    return 0;
}
