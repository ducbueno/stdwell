cmake_minimum_required(VERSION "3.16.3")

project("stdwell")

if(NOT CMAKE_DISABLE_FIND_PACKAGE_CUDA AND
    CMAKE_VERSION VERSION_GREATER 3.7.9)
  if(CMAKE_BUILD_TYPE)
    set(_flags_suffix "_${CMAKE_BUILD_TYPE}")
  endif()
  if(NOT DEFINED ENV{CUDAHOSTCXX} AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND
      (NOT CMAKE_CUDA_FLAGS${_flags_suffix} OR NOT CMAKE_CUDA_FLAGS${_flags_suffix} MATCHES ".*-ccbin .*"))
    message(STATUS "Setting CUDA host compiler CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CXX_COMPILER} to "
      "prevent incompatibilities. Note that this might report that there "
      "is not CUDA compiler if your system's CUDA compiler does not support "
      "${CMAKE_CXX_COMPILER}.")
    # check_language does not seem to care about ${CMAKE_CUDA_FLAGS} or $(CUDA_NVCC_FLAGS}.
    # Hence we set CMAKE_CUDA_HOST_COMPILER to our C++ compiler.
    # In check_language(CUDA) we will get an error if we in addition put
    # "-ccbin ${CMAKE_CXX_COMPILER}" into CMAKE_CUDA_FLAGS. It results
    # in "${NVCC} -ccbin=${CMAKE_CXX_COMPILER}   -ccbin ${CMAKE_CXX_COMPILER}"
    # which causes nvcc to abort
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(ENV{CUDAHOSTCXX} ${CMAKE_CUDA_HOST_COMPILER}) # The only thing honored by check_language(CUDA)!
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # OPTIONAL is ignored. Hence the magic above to check whether enabling CUDA works
    enable_language(CUDA OPTIONAL)
    # While the documentation says that it is deprecated, FindCUDA seems the
    # only easy way to determine the cublas and cusparse libraries.
    # Hence we call it unconditionally
    # The WellContributions kernel uses __shfl_down_sync, which was introduced in CUDA 9.0
    find_package(CUDA)
  endif()
  if(CUDA_FOUND AND CUDA_VERSION VERSION_LESS "9.0")
    set(CUDA_FOUND OFF)
    message(WARNING "Deactivating CUDA as we require version 9.0 or newer."
      " Found only CUDA version ${CUDA_VERSION}.")
  endif()
endif()

if(CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_executable("${PROJECT_NAME}" "src/kernel.cu" "src/main.cpp")

if(CUDA_FOUND)
  target_link_libraries( "${PROJECT_NAME}" PUBLIC ${CUDA_cusparse_LIBRARY} )
  target_link_libraries( "${PROJECT_NAME}" PUBLIC ${CUDA_cublas_LIBRARY} )
endif()
