cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set( CMAKE_VERBOSE_MAKEFILE on )
project(Simple VERSION 1.0.0 LANGUAGES CXX)
include(FetchContent)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin64)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: None Debug Release RelWithDebInfo
MinSizeRel."
       FORCE )
endif()

set(CMAKE_BUILD_TYPE Release)
set(CXX_RELEASE_FLAGS = -O3 -march=native )

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "*** Have SIZEOF void * = 8, so 64-bit")
    set( IS_64_BIT 1 )
else ()
    message(STATUS "*** SIZEOF void * != 8, so not 64-bit")
    message(FATAL_ERROR "Only 64-bit i supported")
endif ()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_find")
set(PYTORCH_VERSION "1.6.0")
find_package(Torch ${PYTORCH_VERSION} EXACT PATHS "${CMAKE_SOURCE_DIR}/_deps/libtorch")
if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(download_libtorch)
endif()
if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake_find")
    find_package(Torch REQUIRED PATHS "${CMAKE_SOURCE_DIR}/../../../libtorch")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    # There is some bug in CMake (FindCUDA.cmake) that causes it to mess up the -std=c++11 option
    # if you let it propagate it to nvcc in some cases.
    if(NOT MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

FetchContent_Declare(  libpng
        GIT_REPOSITORY      https://github.com/glennrp/libpng.git
        GIT_TAG             master
        UPDATE_DISCONNECTED TRUE
        STEP_TARGETS        update
        )
FetchContent_GetProperties(libpng)
if (NOT libpng_POPULATED)
    FetchContent_Populate(libpng)
    add_subdirectory("${libpng_SOURCE_DIR}" ${libpng_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
FetchContent_MakeAvailable(libpng)
message(STATUS "Downloading libpng - done")

message(STATUS "libpng_BINARY_DIR= ${libpng_BINARY_DIR}")
message(STATUS "libpng_SOURCE_DIR= ${libpng_SOURCE_DIR}")
# png.h(330): fatal error C1083: Cannot open include file: 'pnglibconf.h'
#/* If pnglibconf.h is missing, you can * copy scripts/pnglibconf.h.prebuilt to pnglibconf.h
file(COPY "${libpng_SOURCE_DIR}/scripts/pnglibconf.h.prebuilt" DESTINATION "${libpng_SOURCE_DIR}")
configure_file("${libpng_SOURCE_DIR}/pnglibconf.h.prebuilt" "${libpng_SOURCE_DIR}/pnglibconf.h" COPYONLY)


# TARGET
SET( PROJ_SOURCES_DIR   src)
SET( PROJ_INCLUDES  include)
SET( PROJ_HEADERS  ${PROJ_INCLUDES}/utils/vision_utils.hpp)
set(PROJECT_INCLUDE_DIR ${PROJ_INCLUDES} )
SET(PROJECT_SOURCE_DIR ${PROJ_SOURCES})

add_executable(${PROJECT_NAME} ${PROJ_SOURCES} ${PROJ_HEADERS} ${PROJECT_INCLUDE_DIR})
target_sources(${PROJECT_NAME} PRIVATE src/neural.cpp) # main entry point is here
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES}  ${OpenCV_LIBRARIES} "${CMAKE_CURRENT_LIST_DIR}/lib/libpng/libpng16.lib" "${CMAKE_CURRENT_LIST_DIR}/lib/zlib/zlib.lib")
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJ_INCLUDES} PUBLIC ${libpng_SOURCE_DIR} )

set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)

set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}") # siv needs this
endforeach()

if(MSVC)    
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif(MSVC)

# # Copy trained PTH files
file(COPY "resources/style_model_cpp.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/erfnet_fs.pt" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)
file(COPY "resources/clion.png" DESTINATION ${CMAKE_SOURCE_DIR}/bin64)

message(STATUS "----------------- INSTALL LOCATIONS -----------------")
message(STATUS "CMAKE_INSTALL_PREFIX          = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_BINARY_DIR              = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_INSTALL_LIBDIR          = ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "INSTALL_CMAKE_DIR             = ${INSTALL_CMAKE_DIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR      = ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "-----------------------------------------------------")

#set(Siv3D_DIR c:/Users/gpu/OpenSiv3D/lib/Windows)
#target_link_libraries(${PROJECT_NAME}  ${Siv3D_DIR} ${Siv3D_DIR}/Siv3D.lib)
#target_link_libraries(${PROJECT_NAME}  ${Siv3D_DIR}/zlib/zlib.lib)
#SET( PROJ_SOURCES   ${PROJ_SOURCES_DIR}/siv.cpp)
#SET( PROJ_HEADERS  ${PROJ_INCLUDES}/Siv3D.hpp)
#target_include_directories(${PROJECT_NAME} PUBLIC ${PROJ_INCLUDES}/Siv3D.hpp} PUBLIC ${PROJ_INCLUDES} PUBLIC ${PROJ_INCLUDES}/ThirdParty/)