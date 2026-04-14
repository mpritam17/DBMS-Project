cmake_minimum_required(VERSION 3.16)
project(dbms_term_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(storage_manager
    cpp/src/storage_manager.cpp
    cpp/src/vec1_reader.cpp
    cpp/src/slotted_page.cpp
    cpp/src/rtree_node.cpp
    cpp/src/rtree_index.cpp
    cpp/src/lru_replacer.cpp
    cpp/src/buffer_pool_manager.cpp
)

target_include_directories(storage_manager
    PUBLIC cpp/include
)

add_executable(storage_smoke
    cpp/tests/storage_manager_smoke.cpp
)

target_link_libraries(storage_smoke
    PRIVATE storage_manager
)

add_executable(bulk_load
    cpp/tools/bulk_load.cpp
)

target_link_libraries(bulk_load
    PRIVATE storage_manager
)

add_executable(packing_test
    cpp/tests/packing_test.cpp
)

target_link_libraries(packing_test
    PRIVATE storage_manager
)

add_executable(bpm_test
    cpp/tests/bpm_test.cpp
)

target_link_libraries(bpm_test
    PRIVATE storage_manager
)

add_executable(bpm_benchmark
    cpp/tests/bpm_benchmark.cpp
)

target_link_libraries(bpm_benchmark
    PRIVATE storage_manager
)

add_executable(rtree_node_test
    cpp/tests/rtree_node_test.cpp
)

target_link_libraries(rtree_node_test
    PRIVATE storage_manager
)

add_executable(rtree_insert_test
    cpp/tests/rtree_insert_test.cpp
)

target_link_libraries(rtree_insert_test
    PRIVATE storage_manager
)

add_executable(rtree_knn_test
    cpp/tests/rtree_knn_test.cpp
)

target_link_libraries(rtree_knn_test
    PRIVATE storage_manager
)

add_executable(rtree_point_test
    cpp/tests/rtree_point_test.cpp
)

target_link_libraries(rtree_point_test
    PRIVATE storage_manager
)

add_executable(rtree_metadata_test
    cpp/tests/rtree_metadata_test.cpp
)

target_link_libraries(rtree_metadata_test
    PRIVATE storage_manager
)

add_executable(week4_query_benchmark
    cpp/tools/week4_query_benchmark.cpp
)

target_link_libraries(week4_query_benchmark
    PRIVATE storage_manager
)

