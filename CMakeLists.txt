# project
cmake_minimum_required (VERSION 2.6)
project (SfM)

cmake_policy(SET CMP0015 NEW)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
add_definitions(-DUSE_SIMD)

function(src2strmap NAME DIR OUT_FILE)
file(GLOB SHADER_SRC_FILES ${DIR}/*)
add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND src2strmap ${CMAKE_CURRENT_SOURCE_DIR}/${DIR} ${NAME} > ${OUT_FILE}
    DEPENDS ${SHADER_SRC_FILES}
)
endfunction(src2strmap)

function(cl2aocxmap NAME DIR OUT_FILE)
file(GLOB SHADER_SRC_FILES ${DIR}/*)
add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND cl2aocxmap ${CMAKE_CURRENT_SOURCE_DIR}/${DIR} ${NAME} ${OUT_FILE} -fp-relaxed=true
    DEPENDS ${SHADER_SRC_FILES}
)
endfunction(cl2aocxmap)

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm)
# arm
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a9 -mfpu=neon")
    include_directories("/opt/altera/13.1/hld/host/include/")
    include_directories("../extern/root/include")
    link_directories("../extern/root/lib")
    link_directories("/opt/gcc-linaro-arm-linux-gnueabihf-4.8-2014.04_linux/arm-linux-gnueabihf/lib/arm-linux-gnueabi/")
    link_directories("/opt/altera/13.1/hld/board/c5soc/arm32/lib/")
    list(APPEND extra_libs alteracl elf alterahalmmd alterammdpcie c_accel_runtime stdc++ rt)

    src2strmap(SHADER_SRCS demo/shaders shader_srcs.generated.hh)
    cl2aocxmap(KERNEL_SRCS viso/kernels kernel_srcs.generated.hh)
else()
# x86
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    list(APPEND extra_libs sfml-graphics sfml-window sfml-system GLEW GL OpenCL)
    
    src2strmap(SHADER_SRCS demo/shaders shader_srcs.generated.hh)
    src2strmap(KERNEL_SRCS viso/kernels kernel_srcs.generated.hh)
endif()

include_directories("viso")
include_directories("demo")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB SRC_FILES "viso/*.cpp")
add_library(viso2 ${SRC_FILES};kernel_srcs.generated.hh)

file(GLOB SRC_FILES "demo/*.cc")
add_library(sfmdemo ${SRC_FILES};shader_srcs.generated.hh)
target_link_libraries(sfmdemo viso2 png ${extra_libs})

add_executable(sfm main.cpp)
target_link_libraries(sfm sfmdemo)

add_executable(simd_test test/simd.cpp)
target_link_libraries(simd_test sfmdemo)

