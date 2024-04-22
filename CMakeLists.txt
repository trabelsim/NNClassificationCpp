project(NNGen)
cmake_minimum_required(VERSION 3.12)

set(SOURCES
    src/NNHelper.cpp
    src/NNLayer.cpp
    src/NNInput.cpp
    src/main.cpp)


set(HEADER_FILES 
    include/NNLayer.h
    include/NNHelper.h
    include/NNInput.h )


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADER_FILES})

target_include_directories( ${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include)