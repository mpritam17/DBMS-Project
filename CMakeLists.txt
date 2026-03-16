cmake_minimum_required(VERSION 3.16)
project(dbms_term_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(storage_manager
    cpp/src/storage_manager.cpp
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
