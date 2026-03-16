# Week 3 Index Foundation

This milestone adds the first R-tree specific storage layer on top of the existing 4KB page abstraction and extends it with basic insertion.

## Implemented

- `BoundingBox` math for high-dimensional minimum bounding rectangles.
- `RTreeNodePage`, a serializable node wrapper over a raw 4096-byte page.
- Support for both leaf and internal node page types.
- Entry serialization as `lower_bounds + upper_bounds + uint64_t value`.
- Metadata for parent-page links and leaf-chain links.
- Automated round-trip serialization test.
- `RTreeIndex`, which inserts records recursively, splits overflowing nodes, and grows a new root when needed.
- Automated insertion test that forces multi-page splits and validates the resulting tree contents.
- `RTreeIndex::searchKNN`, a branch-and-bound KNN search that traverses the R-tree via the buffer pool, pruning subtrees whose minimum MBR distance to the query exceeds the current k-th nearest result.
- Automated KNN test covering 2D structured queries, brute-force comparison on random data, and 128-dimensional point recovery.

## Page layout

Each R-tree node uses one 4KB page:

- Bytes `0..63`: shared `PageHeader`
- Bytes `64..95`: `RTreeNodePage::NodeHeader`
- Remaining bytes: fixed-width R-tree entries

For a node with `d` dimensions, each entry occupies:

`entry_bytes = 2 * d * sizeof(float) + sizeof(uint64_t)`

At `d = 128`, each entry is `1032` bytes, so a 4KB page fits `3` entries.

## Validation

Build and run:

```bash
cmake -S . -B build
cmake --build build
./build/rtree_node_test
./build/rtree_insert_test
./build/rtree_knn_test
```

Expected result:

- The test verifies node round-trip serialization.
- The test verifies node MBR aggregation.
- The test verifies `128`-dimensional page capacity is exactly `3` entries.
- The insertion test verifies recursive inserts, page splits, root growth, and preservation of all inserted values.
- The KNN test verifies nearest-neighbour distances match brute-force results across 2D and 128D data.

## What comes next

The current insertion and search path does the following:

1. Chooses a subtree by minimum MBR enlargement (insertion).
2. Inserts into leaf pages, splits overflowing nodes, and grows a new root.
3. Propagates updated MBRs and child links back to the root.
4. Searches KNN via branch-and-bound: maintains a min-heap of subtrees ordered by minimum distance to the query, pruning whenever the lower bound exceeds the current k-th distance.

Current limitation:

- Root page metadata still lives in the `RTreeIndex` object, not in a dedicated on-disk metadata page. The tree is page-backed, but reopening an existing index cleanly is a follow-up task.
