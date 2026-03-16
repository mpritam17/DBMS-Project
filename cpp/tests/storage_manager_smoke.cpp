#include "storage_manager.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
    const std::string db_file = "week1_smoke.db";

    try {
        if (std::filesystem::exists(db_file)) {
            std::filesystem::remove(db_file);
        }

        StorageManager storage;
        storage.open(db_file);

        const auto page_id = storage.allocatePage();
        std::vector<std::uint8_t> page(StorageManager::kPageSize, 0);

        page[0] = 0xDB;
        page[1] = 0x5A;
        page[2] = 0x01;

        storage.writePage(page_id, page);
        const auto read_back = storage.readPage(page_id);

        if (read_back[0] != 0xDB || read_back[2] != 0x01) {
            std::cerr << "Smoke test failed: unexpected page content\n";
            return 1;
        }

        storage.flush();
        storage.close();

        std::cout << "StorageManager smoke test passed. Pages: 1\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Smoke test failed with exception: " << ex.what() << "\n";
        return 1;
    }
}
