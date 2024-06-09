project(NNGen)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

# Find Catch2 package
find_package(Catch2 REQUIRED)

# Include directories
include_directories(include)

set(SOURCES
    src/NNHelper.cpp
    src/NNLayer.cpp
    src/NNInput.cpp
    src/NNLoss.cpp
    src/NNActivation.cpp
    src/NNAccuracy.cpp
    src/Optimizers/SGD.cpp
    src/main.cpp)


set(HEADER_FILES 
    include/NNLayer.h
    include/NNHelper.h
    include/NNInput.h 
    include/NNLoss.h
    include/NNAccuracy.h
    include/NNActivation.h
    include/Optimizers/BaseOptimizer.h
    include/Optimizers/SGD.h)


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADER_FILES})

target_include_directories( ${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/Optimizers)

#TESTS

include_directories(tests/config/)

set(TEST_SOURCES_NN_HELPER tests/Helper/test.cpp src/NNHelper.cpp)
add_executable(testHelper ${TEST_SOURCES_NN_HELPER})
target_link_libraries(testHelper PUBLIC Catch2::Catch2WithMain)

set(TEST_SOURCES_NN_LAYER tests/Layer/test.cpp src/NNLayer.cpp src/NNHelper.cpp)
add_executable(testLayer ${TEST_SOURCES_NN_LAYER})
target_link_libraries(testLayer PUBLIC Catch2::Catch2WithMain)

set(TEST_SOURCES_NN_LOSS tests/Loss/test.cpp src/NNLoss.cpp src/NNHelper.cpp src/NNActivation.cpp)
add_executable(testLoss ${TEST_SOURCES_NN_LOSS})
target_link_libraries(testLoss PUBLIC Catch2::Catch2WithMain)