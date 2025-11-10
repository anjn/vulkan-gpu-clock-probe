cmake_minimum_required(VERSION 3.20)
project(clock-probe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/alu.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/alu.comp.spv)

add_custom_command(
OUTPUT ${SHADER_SPV}
COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute ${SHADER_SRC} -o ${SHADER_SPV}
DEPENDS ${SHADER_SRC}
COMMENT "Compiling GLSL -> SPIR-V: alu.comp"
)

add_custom_target(Shaders DEPENDS ${SHADER_SPV})

set(TARGETS clock-probe)

foreach(TARGET ${TARGETS})
    add_executable(${TARGET} src/main.cpp ${SHADER_SRC})
    add_dependencies(${TARGET} Shaders)
    target_include_directories(${TARGET} PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${Vulkan_LIBRARIES})
    # Embed SPIR-V path as a compile definition for simplicity
    target_compile_definitions(${TARGET} PRIVATE SHADER_SPV_PATH="${SHADER_SPV}")

    #    add_executable(
    #        ${TARGET}
    #        ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cpp
    #        ${AMF_SDK_SOURCES}
    #    )
    #
    #    target_include_directories(
    #        ${TARGET} PRIVATE
    #        ${AMF_SDK_DIR}/amf
    #        ${Vulkan_INCLUDE_DIRS}
    #    )
    #
    #    target_link_libraries(
    #        ${TARGET}
    #        Vulkan::Vulkan
    #    )
endforeach(TARGET)
