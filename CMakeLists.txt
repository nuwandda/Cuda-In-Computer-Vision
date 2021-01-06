cmake_minimum_required(VERSION 2.8)
PROJECT(Cuda-In-Computer-Vision)
find_package(OpenCV REQUIRED)

set(NAME_SRC
        main.cpp
        src/filters/cpu/filters.cpp
        src/detectors/edge/cpu/edge_detector.cpp)

set(NAME_HEADERS
        include/filters/cpu/filters.h
        include/detectors/edge/cpu/edge_detector.h include/convert/rgb_to_gray.h src/convert/cpu/rgb_to_gray.cpp)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(Cuda-In-Computer-Vision ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries(Cuda-In-Computer-Vision ${OpenCV_LIBS})