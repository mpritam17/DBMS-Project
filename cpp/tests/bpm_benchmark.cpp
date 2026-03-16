#include "buffer_pool_manager.h"
#include "storage_manager.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>

const int NUM_HOT_THREADS = 4;
const int NUM_OPERATIONS = 10000;
const int POOL_SIZE = 50; 
const int DB_PAGES = 500;

// Hot threads simulate highly repeated lookups on very specific index nodes (pages 0 to 10)
void hot_access_worker(BufferPoolManager* bpm, int thread_id) {
    std::mt19937 rng(thread_id);
    std::uniform_int_distribution<uint32_t> dist(0, 10);
    
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        uint32_t page_id = dist(rng);
        Page* page = bpm->fetchPage(page_id);
        
        if (page) {
            page->RLock();
            // simulate fast memory read
            page->RUnlock();
            bpm->unpinPage(page_id, false);
        }
    }
}

// Sequential scan simulates a full table scan that forces the LRU to evict everything
void sequential_scan_worker(BufferPoolManager* bpm) {
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        uint32_t page_id = i % DB_PAGES;
        Page* page = bpm->fetchPage(page_id);
        if (page) {
            bpm->unpinPage(page_id, false);
        }
    }
}

int main() {
    const std::string db_file = "benchmark.db";
    std::remove(db_file.c_str());
    
    // Reset global metrics
    StorageManager::disk_reads = 0;
    StorageManager::disk_writes = 0;

    // Setup initial database pages so they actually exist on disk
    {
        StorageManager disk_manager;
        disk_manager.open(db_file);
        for(int i=0; i<DB_PAGES; i++) disk_manager.allocatePage();
        disk_manager.close();
    }

    {
        StorageManager disk_manager;
        disk_manager.open(db_file);
        BufferPoolManager bpm(POOL_SIZE, &disk_manager);

        std::cout << "Starting BPM Performance Benchmark..." << std::endl;
        std::cout << "Buffer Pool Size: " << POOL_SIZE << " frames" << std::endl;
        std::cout << "Database Size: " << DB_PAGES << " pages" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        
        // Spawn Hot index lookup threads
        for (int i = 0; i < NUM_HOT_THREADS; ++i) {
            threads.emplace_back(hot_access_worker, &bpm, i);
        }
        
        // Spawn sequential scanner
        threads.emplace_back(sequential_scan_worker, &bpm);
        
        for (auto& t : threads) {
            t.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Total Time Taken: " << elapsed_ms << " ms" << std::endl;
        std::cout << "Disk Reads (Penalties): " << StorageManager::disk_reads << std::endl;
        std::cout << "Disk Writes (Penalties): " << StorageManager::disk_writes << std::endl;
        std::cout << "Cache Hit Ratio effect can be seen directly via disk IO." << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }
    
    std::remove(db_file.c_str());
    return 0;
}
