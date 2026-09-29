cmake_minimum_required(VERSION 3.27)
project(MonteCarloOptionPricingProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_ROOT "/usr/local/boost")
find_package(Boost REQUIRED CONFIG)

set(Eigen3_DIR "/usr/local/eigen3/share/eigen3/cmake")
find_package(Eigen3 REQUIRED CONFIG)

add_executable(PE PricingEngine.hpp PricingEngine.cpp main.cpp)

target_include_directories(PE PRIVATE 
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(PE PRIVATE 
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)