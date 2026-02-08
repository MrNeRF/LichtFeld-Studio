# This CMakeLists only does find_package to test our finders.
# We might want to enable some calls based on what is actually installed on the CI runner.


cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(CMakeLibraryCITest)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../finders")

find_package(GMM)
find_package(Gurobi)
find_package(Mosek)


enable_testing()
add_subdirectory(tests)

