# Buffer Pool Manager Performance Results

This file documents the actual benchmark outputs for the Week 2 buffer pool changes. The previous version of this document overstated the outcome. The measured data shows that the LRU-2 style replacer reduced disk reads under scan pressure, but the saved runs do not show a consistent wall-clock speedup.

## What changed

The optimized implementation combines two changes:

1. LRU-2 style replacement in [cpp/src/lru_replacer.cpp](/home/quantumec/Documents/DBMS_term_project/cpp/src/lru_replacer.cpp), where first-touch pages stay in a FIFO queue and only repeatedly used pages move into the hot LRU queue.
2. Reduced global lock hold time in [cpp/src/buffer_pool_manager.cpp](/home/quantumec/Documents/DBMS_term_project/cpp/src/buffer_pool_manager.cpp), where disk I/O happens outside the main BPM latch.

Because both changes shipped together, the benchmark below is a before/after comparison of the whole optimization set, not an isolated measurement of LRU-2 alone.

## Benchmark workload

The benchmark in [cpp/tests/bpm_benchmark.cpp](/home/quantumec/Documents/DBMS_term_project/cpp/tests/bpm_benchmark.cpp) runs:

- Buffer pool size: 50 frames
- Database size: 500 pages
- 4 hot threads repeatedly reading pages 0 to 10
- 1 sequential scan thread reading the full page range repeatedly

This workload is intended to test whether scan traffic pollutes the cache and evicts frequently reused hot pages.

## Saved results

The repository already contains three benchmark artifacts:

| Run | Source | Total Time (ms) | Disk Reads | Delta vs baseline time | Delta vs baseline reads |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | [baseline.txt](/home/quantumec/Documents/DBMS_term_project/baseline.txt) | 202.829 | 10294 | - | - |
| Optimized | [optimized.txt](/home/quantumec/Documents/DBMS_term_project/optimized.txt) | 243.666 | 9794 | +40.837 ms (+20.1%) | -500 (-4.86%) |
| Optimized final | [optimized_final.txt](/home/quantumec/Documents/DBMS_term_project/optimized_final.txt) | 290.541 | 9791 | +87.712 ms (+43.2%) | -503 (-4.89%) |

## Current reruns

I reran the current benchmark three times on 17 March 2026:

| Run | Total Time (ms) | Disk Reads |
| --- | ---: | ---: |
| Current run 1 | 165.583 | 9791 |
| Current run 2 | 224.990 | 9791 |
| Current run 3 | 288.163 | 9791 |
| Average | 226.245 | 9791 |
| Median | 224.990 | 9791 |

Compared with the saved baseline artifact, the current optimized code still shows the same read count reduction of 503 reads, which is a 4.89% drop in disk-read penalties. Using the three reruns above, average elapsed time is 23.416 ms slower than the saved baseline, which is an 11.5% regression in wall-clock time.

## What the results actually say

The current data supports these conclusions:

- The replacer change is doing its intended job of protecting the hot set better than the old policy. The repeatable signal is lower disk-read count: 10294 down to 9791.
- The saved and rerun timings do not justify the claim that the optimized implementation is faster overall. In the artifacts checked into the repo, elapsed time got worse even though read misses decreased.
- The benchmark is high-variance. Current reruns ranged from 165.583 ms to 288.163 ms, which means a single run is not enough to make a strong speed claim.

## Why time and read-count disagree

There are several likely reasons:

- The benchmark mixes two effects together: replacement policy and lock restructuring.
- It measures total runtime across five threads, so scheduler noise and lock contention can dominate a small reduction in disk reads.
- The disk-read counter measures cache behavior directly, which is more closely tied to the LRU-2 change than total elapsed time is.
- A 4.89% drop in reads may simply be too small to overcome synchronization overhead in this particular test setup.

## Bottom line

The honest summary is:

- LRU-2 improved eviction behavior under the benchmarked mixed workload.
- The measured before/after data in this repository does not yet prove an end-to-end throughput win.
- If we want to claim a real performance improvement, we need a better benchmark that separates replacer effects from concurrency effects and reports averages over many runs.