cmake_minimum_required(VERSION 3.1...3.23)
project(AlphaShaper)

# MAKE SURE TO INSTALL THE CGAL LIBRARY AND ITS DEPENDENCIES
find_package(CGAL REQUIRED)
#cmake_policy(SET CMP0167 NEW)
#find_package(Boost REQUIRED)

add_executable(AlphaShaper main.cpp)

target_link_libraries(AlphaShaper CGAL::CGAL)