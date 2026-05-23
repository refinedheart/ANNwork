cmake_minimum_required(VERSION 3.20)
project(ann_x86_win LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(ann_x86_win
    main_x86_win.cc
)

target_include_directories(ann_x86_win PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ann_x86_win PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(ann_x86_win PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(ann_x86_win PRIVATE -O2 -mavx2 -mfma -msse4.1)
endif()
