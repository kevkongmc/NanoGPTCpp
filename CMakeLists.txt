cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(nano_gpt VERSION 0.1.0 LANGUAGES CXX)

# Enable compilation database generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to local LibTorch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")

# Find LibTorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/third_party/abseil-cpp
)

# Add Abseil subdirectory
add_subdirectory(third_party/abseil-cpp)

# Add source files
set(SOURCES
    nano_gpt_main.cc
    tokenizer.cc
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add executables
add_executable(example example.cc)
add_executable(nano_gpt ${SOURCES})

# Link libraries to the executables
target_link_libraries(example PRIVATE "${TORCH_LIBRARIES}")
# Include Abseil libraries for convenience
set(ABSL_LIBS
    # Core libraries
    absl::flags
    absl::strings
    absl::status
    absl::statusor
    
    # Container libraries
    absl::any
    absl::btree
    absl::fixed_array
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
    absl::node_hash_map
    absl::node_hash_set
    absl::synchronization
    
    # Logging libraries
    absl::log
    absl::log_initialize
    absl::log_flags
    absl::log_globals
)

# Link libraries to the executables
target_link_libraries(example PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(nano_gpt PRIVATE 
    "${TORCH_LIBRARIES}"
    ${ABSL_LIBS}
)

# Enable C++17 and set properties for all targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile features for each target
target_compile_features(example PRIVATE cxx_std_17)
target_compile_features(nano_gpt PRIVATE cxx_std_17)

# Set the directory where the executables will be built (project root)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
