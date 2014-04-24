# project
cmake_minimum_required (VERSION 2.6)
project (SfM)

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3 -std=c++11")
# make release version
set(CMAKE_BUILD_TYPE Debug)

# include directory
include_directories("viso")
include_directories("demo")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

function(src2strmap NAME DIR OUT_FILE)
file(GLOB SHADER_SRC_FILES ${DIR}/*)
add_custom_command(
    OUTPUT ${OUT_FILE}
    COMMAND src2strmap ${CMAKE_CURRENT_SOURCE_DIR}/${DIR} ${NAME} > ${OUT_FILE}
    DEPENDS ${SHADER_SRC_FILES}
)
endfunction(src2strmap)

src2strmap(SHADER_SRCS demo/shaders shader_srcs.generated.hh)
src2strmap(KERNEL_SRCS viso/kernels kernel_srcs.generated.hh)

file(GLOB SRC_FILES "viso/*.cpp")
add_library(viso2 ${SRC_FILES};kernel_srcs.generated.hh)

file(GLOB SRC_FILES "demo/*.cc")
add_library(sfm_demo ${SRC_FILES};shader_srcs.generated.hh)
target_link_libraries(sfm_demo viso2 png sfml-graphics sfml-window sfml-system GLEW GL OpenCL)

file(GLOB srcs "./*.cpp")
foreach(src_file ${srcs})
    get_filename_component(demo_name ${src_file} NAME_WE)
    message(${src_file})
    add_executable(${demo_name} ${src_file})
    target_link_libraries(${demo_name} sfm_demo)
endforeach(src_file ${srcs})

