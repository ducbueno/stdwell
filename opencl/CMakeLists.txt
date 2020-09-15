cmake_minimum_required(VERSION "3.16.3")

project("stdwell")

find_package(OpenCL)

if(OpenCL_FOUND)
  find_file(CL_HPP CL/cl.hpp HINTS ${OpenCL_INCLUDE_DIRS})
  if(CL_HPP)
    include_directories(${OpenCL_INCLUDE_DIRS})
  else()
    message(WARNING "OpenCL was found, but not the required version (1.2)")
  endif()
endif()

add_executable("${PROJECT_NAME}" "src/opencl.cpp" "src/main.cpp")

if(OpenCL_FOUND)
  target_link_libraries("${PROJECT_NAME}" ${OpenCL_LIBRARIES})
endif()
