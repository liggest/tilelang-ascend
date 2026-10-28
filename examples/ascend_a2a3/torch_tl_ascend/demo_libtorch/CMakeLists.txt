cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attention_demo LANGUAGES CXX)

# Preserve the original standalone libtorch_npu development dependency. The
# Python TorchNPU wheel alone does not provide its installed CMake package/API.
if(NOT DEFINED TORCH_NPU_SOURCE_DIR OR NOT IS_DIRECTORY "${TORCH_NPU_SOURCE_DIR}")
    message(FATAL_ERROR "TORCH_NPU_SOURCE_DIR must name the matching real torch_npu source root")
endif()

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(flash_attention_demo flash_attention.cpp)

target_include_directories(flash_attention_demo PRIVATE ${TORCH_NPU_SOURCE_DIR})

target_link_libraries(flash_attention_demo PUBLIC "${TORCH_LIBRARIES}" "${TORCH_NPU_LIBRARIES}")
target_link_libraries(flash_attention_demo PRIVATE dl)  # support dlopen

set_target_properties(flash_attention_demo PROPERTIES
    CXX_STANDARD 17
    BUILD_RPATH "$ORIGIN;$ORIGIN/.."
    INSTALL_RPATH "$ORIGIN"
)
