// rtree_metadata_test.cpp
// Verifies that RTreeIndex persists its metadata (root page id, dimensions,
// height) to disk so that an existing tree can be reopened by meta_page_id
// after the original process has closed.
//
// Test flow:
//  1. Create a fresh index, insert N 2D points.
//  2. Flush all BPM pages to disk, close the storage file.
//  3. Reopen the same storage file with a brand-new StorageManager + BPM.
//  4. Construct RTreeIndex from meta_page_id (open path).
//  5. Verify dimensions, height, root_page_id are consistent.
//  6. Run the same KNN query and confirm the distances match the original run.

#include "buffer_pool_manager.h"
#include "rtree_index.h"
#include "rtree_node.h"
#include "storage_manager.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char* kDbFile = "/tmp/rtree_metadata_test.db";

int main() {
    std::remove(kDbFile);

    // -----------------------------------------------------------------------
    // Phase 1: create, populate, and flush.
    // -----------------------------------------------------------------------
    uint32_t saved_meta_page_id;
    uint32_t saved_root_page_id;
    uint16_t saved_dims;
    std::size_t saved_height;
    std::vector<std::pair<float, uint64_t>> original_knn;

    {
        StorageManager sm;
        sm.open(kDbFile);
        BufferPoolManager bpm(64, &sm);
        RTreeIndex idx(&bpm, static_cast<uint16_t>(2));

        // Insert 15 2D points: (i*5, i*3) with value = i.
        for (uint64_t i = 0; i < 15; ++i) {
            idx.insertPoint({static_cast<float>(i * 5), static_cast<float>(i * 3)}, i);
        }

        // Run a KNN query before closing.
        original_knn = idx.searchKNN({20.0f, 12.0f}, 3);
        assert(original_knn.size() == 3 && "expected 3 KNN results from original index");

        // Snapshot the metadata that the opener will need.
        saved_meta_page_id = idx.getMetaPageId();
        saved_root_page_id = idx.getRootPageId();
        saved_dims         = idx.getDimensions();
        saved_height       = idx.getHeight();

        assert(saved_meta_page_id != std::numeric_limits<uint32_t>::max() &&
               "meta_page_id must be valid");

        // Flush everything to disk. The BPM and SM destructors will also
        // clean up correctly when the scope ends.
        bpm.flushAllPages();
    }

    // -----------------------------------------------------------------------
    // Phase 2: reopen from disk, verify metadata, run same KNN query.
    // -----------------------------------------------------------------------
    {
        StorageManager sm;
        sm.open(kDbFile);
        BufferPoolManager bpm(64, &sm);

        // Open the existing index by meta page id (no dimensions argument).
        RTreeIndex idx(&bpm, saved_meta_page_id);

        assert(idx.getDimensions() == saved_dims &&
               "reopened index must have same dimensions");
        assert(idx.getHeight() == saved_height &&
               "reopened index must have same height");
        assert(idx.getRootPageId() == saved_root_page_id &&
               "reopened index must have same root page id");
        assert(idx.getMetaPageId() == saved_meta_page_id &&
               "reopened index must have same meta page id");

        // Run the same KNN query on the reopened tree.
        auto reopened_knn = idx.searchKNN({20.0f, 12.0f}, 3);
        assert(reopened_knn.size() == 3 && "expected 3 KNN results from reopened index");

        for (std::size_t i = 0; i < 3; ++i) {
            assert(std::abs(reopened_knn[i].first - original_knn[i].first) < 1e-3f &&
                   "KNN distance mismatch after reopen");
            assert(reopened_knn[i].second == original_knn[i].second &&
                   "KNN value mismatch after reopen");
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: error handling — bad meta page id must throw.
    // -----------------------------------------------------------------------
    {
        StorageManager sm;
        sm.open(kDbFile);
        BufferPoolManager bpm(16, &sm);

        bool threw = false;
        try {
            // Pass the root page id as if it were the meta page id — magic check fails.
            RTreeIndex bad_idx(&bpm, saved_root_page_id);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw && "opening index from a non-meta page must throw");
    }

    std::remove(kDbFile);
    std::printf("R-tree metadata persistence test passed.\n");
    return 0;
}
