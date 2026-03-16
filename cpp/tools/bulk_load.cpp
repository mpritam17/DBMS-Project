#include "vec1_reader.h"
#include "slotted_page.h"
#include "storage_manager.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_vec1.bin> <output_db_file>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    try {
        Vec1Reader reader(input_file);
        const auto& header = reader.getHeader();
        std::cout << "Loading VEC1 file:\n";
        std::cout << "  Count: " << header.count << "\n";
        std::cout << "  Dims: " << header.dims << "\n";

        StorageManager storage;
        storage.open(output_file);

        uint32_t current_page_id = storage.allocatePage();
        SlottedPage current_page(current_page_id);
        
        uint64_t total_loaded = 0;
        Vec1Entry entry;
        
        while (reader.readNext(entry)) {
            // Calculate size to write
            size_t entry_bytes = sizeof(entry.id) + entry.vector.size() * sizeof(float);
            
            // Pack it into a temporary buffer for copying
            std::vector<uint8_t> buffer(entry_bytes);
            std::memcpy(buffer.data(), &entry.id, sizeof(entry.id));
            std::memcpy(buffer.data() + sizeof(entry.id), entry.vector.data(), entry.vector.size() * sizeof(float));

            if (!current_page.addItem(buffer.data(), entry_bytes)) {
                // Not enough space: write current page out and allocate a new one
                storage.writePage(current_page_id, current_page.getRawData());
                
                current_page_id = storage.allocatePage();
                current_page = SlottedPage(current_page_id);
                
                // Try adding again
                if (!current_page.addItem(buffer.data(), entry_bytes)) {
                    throw std::runtime_error("Entry is too large to fit in an empty 4KB page!");
                }
            }
            
            total_loaded++;
        }
        
        // Write the last page if it has strictly been modified
        storage.writePage(current_page_id, current_page.getRawData());
        
        storage.flush();
        storage.close();

        std::cout << "Successfully loaded " << total_loaded << " entries into " 
                  << storage.pageCount() << " pages.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
