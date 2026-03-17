# Week 3 Index Foundation

This milestone adds the first R-tree specific storage layer on top of the existing page abstraction and extends it with basic insertion.

## Implemented

- `BoundingBox` math for high-dimensional minimum bounding rectangles.
- `RTreeNodePage`, a serializable node wrapper over a raw 4096-byte page.
- `RTreeNodePage`, a serializable node wrapper over a raw 16384-byte page.
- Support for both leaf and internal node page types.
- Entry serialization as `lower_bounds + upper_bounds + uint64_t value`.
- Metadata for parent-page links and leaf-chain links.
- Automated round-trip serialization test.
- `RTreeIndex`, which inserts records recursively, splits overflowing nodes, and grows a new root when needed.
- Automated insertion test that forces multi-page splits and validates the resulting tree contents.
- `RTreeIndex::searchKNN`, a branch-and-bound KNN search that traverses the R-tree via the buffer pool, pruning subtrees whose minimum MBR distance to the query exceeds the current k-th nearest result.
- Automated KNN test covering 2D structured queries, brute-force comparison on random data, and 128-dimensional point recovery.

## Page layout

Each R-tree node uses one 16KB page:

- Bytes `0..63`: shared `PageHeader`
- Bytes `64..95`: `RTreeNodePage::NodeHeader`
- Remaining bytes: fixed-width R-tree entries

For a node with `d` dimensions, each entry occupies:

`entry_bytes = 2 * d * sizeof(float) + sizeof(uint64_t)`

At `d = 128`, each entry is `1032` bytes, so a 16KB page fits `15` entries.


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
- The test verifies `128`-dimensional page capacity is exactly `15` entries.
- The insertion test verifies recursive inserts, page splits, root growth, and preservation of all inserted values.
- The KNN test verifies nearest-neighbour distances match brute-force results across 2D and 128D data.

## What comes next

## Performance Improvements

### Fix 1 — Page size: 4 KB → 16 KB

**Problem:** At 128 dimensions each R-tree entry occupies 1032 bytes.  A 4 KB page holds only 3 entries, giving a branching factor of 3.  With 50 vectors the tree reaches height ≈ 4 and the branch-and-bound KNN degrades to a near-linear scan because almost every subtree is kept in the priority queue.

| Metric | Before (4 KB page) | After (16 KB page) |
|--------|--------------------|--------------------|
| `kPageSize` | 4096 bytes | 16384 bytes |
| Entries per page at 128D | 3 | 15 |
| Branching factor | 3 | 15 |
| Tree height for 50 vectors | ~4 | ~2 |
| Subtrees pruned per KNN query | few | many (5× improvement) |

The change is a single constant in `PageLayout::kPageSize`.  All existing slotted-page and BPM logic adapts automatically.  Slotted-page capacity also increases (31 items per page vs 7 previously).

### Fix 2 — Log-space subtree selection in `chooseSubtree`

**Problem:** The standard R-tree `chooseSubtree` criterion picks the child whose MBR needs the least volume enlargement to cover the new entry.  The enlargement is computed as:

```
enlargement = hyperVolume(expanded_MBR) - hyperVolume(original_MBR)
```

`hyperVolume` is the product of per-dimension edge lengths.  At 128 dimensions with L2-normalised embeddings (values roughly in `[-3, 3]`), each edge is on the order of `0.01`–`0.1`.  The product of 128 such numbers underflows IEEE 754 `double` to exactly `0.0`.  As a result **every call to `hyperVolume()` returns 0**, the enlargement is always `0 - 0 = 0` for every candidate child, and `chooseSubtree` degenerates to always returning the **first child**.  This produces a heavily unbalanced tree even with the correct split logic in place.

**Fix:** Replace the absolute-volume comparison with a **log-space enlargement ratio**:

```
log_enlargement_ratio(MBR, target) = sum_i  log( expanded_edge_i / orig_edge_i )
								   = log( hyperVolume(expanded) / hyperVolume(original) )
```

Because we are taking logarithms of individual per-dimension ratios (no global product), the result is always a normal `double`.  All relative-enlargement comparisons are preserved: smaller log-ratio means smaller relative growth.  Tie-breaking uses `logHyperVolume()` (a sum of per-dimension `log(edge)`) instead of the underflowing product.

Two new methods were added to `BoundingBox`:

| Method | Description |
|--------|-------------|
| `logHyperVolume()` | Sum of `log(upper[i] - lower[i])`; returns `-inf` for a zero-edge (point) MBR |
| `logEnlargementRatio(other)` | Sum of per-dimension `log(expanded / orig)`; returns `+inf` when a zero-edge must grow |

`chooseSubtree` now calls `logEnlargementRatio` and `logHyperVolume` instead of `enlargementToInclude` and `hyperVolume`.

| Metric | Before (absolute volumes) | After (log-space) |
|--------|--------------------------|-------------------|
| `hyperVolume()` at 128D | underflows to `0.0` | not called in hot path |
| `enlargementToInclude()` at 128D | always `0.0 - 0.0 = 0` | not called in hot path |
| Subtree routed on overflow | always child 0 (unbalanced) | minimum-enlargement child (balanced) |
| Tree quality at 128D | depth-skewed, wide MBR overlap | balanced, minimised MBR overlap |
| Numerical stability | breaks below `~0.01^128 ~ 1e-256` | stable for any positive edge |

---

The current insertion and search path does the following:

1. Chooses a subtree by minimum MBR enlargement (insertion).
2. Inserts into leaf pages, splits overflowing nodes, and grows a new root.
3. Propagates updated MBRs and child links back to the root.
4. Searches KNN via branch-and-bound: maintains a min-heap of subtrees ordered by minimum distance to the query, pruning whenever the lower bound exceeds the current k-th distance.

Current limitation:

- Root page metadata still lives in the `RTreeIndex` object, not in a dedicated on-disk metadata page. The tree is page-backed, but reopening an existing index cleanly is a follow-up task.
