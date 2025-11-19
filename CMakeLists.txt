cmake_minimum_required(VERSION 3.16)
SET(CMAKE_BUILD_TYPE "Release")

set(PROJECT_NAME linear_electrostatic_solver)
project(${PROJECT_NAME})

# find and use packages
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# executable
add_executable(${PROJECT_NAME} main.cpp GPB_cell_model.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC dolfin)


