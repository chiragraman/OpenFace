# @Author: Chirag Raman <chirag>
# @Date:   2016-05-09T18:28:59-04:00
# @Email:  chirag.raman@gmail.com
# @Last modified by:   chirag
# @Last modified time: 2016-05-10T17:33:57-04:00
# @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.

cmake_minimum_required(VERSION 3.5)

# use standard file endings - rename to cxx for source if needed
set( SOURCE_EXT_PATTERN ".cpp" )
set( HEADER_EXT_PATTERN ".h" )

# rename to your name here
set( PROJECT_NAME InMindDemo )

# define project for C++ language
project( ${PROJECT_NAME} CXX )

# set where to find additional cmake modules if any
# comment it out if not required
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} )

# set compiler support for C++11 standard
option( USE_CXX11_STD "Product should be build with C++11 compiler option enabled" ON )

if(USE_CXX11_STD)
    set( CMAKE_CXX_STANDARD 11 )
endif()

# add include path, and any additional paths, for the project
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories(../../lib/local/InmindEmotionDetector/include)
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)

# find dependency packages
# Ffmpeg
find_package( FFmpeg COMPONENTS AVCODEC AVFORMAT AVUTIL AVDEVICE  SWSCALE REQUIRED )

if(FFMPEG_FOUND)
#  FFMPEG_INCLUDE_DIRS  - Include directory necessary for using the required
#                         components headers
#  FFMPEG_LIBRARIES     - Link these to use the required ffmpeg components
#  FFMPEG_DEFINITIONS   - Compiler switches required for using the required
#                         ffmpeg components
    message( "FFMPEG_INCLUDE_DIRS = ${FFMPEG_INCLUDE_DIRS}" )
    message( "FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES}" )
    message( "FFMPEG_DEFINITIONS = ${FFMPEG_DEFINITIONS}" )

    include_directories( ${FFMPEG_INCLUDE_DIRS} )
else()
    message( FATAL_ERROR "FFMPEG not found" )
endif()

#OpenCV
find_package( OpenCV REQUIRED )

#ZMQ
find_package( ZMQ REQUIRED )
if(ZMQ_FOUND AND ZMQ_LIBRARIES)
    message( "ZMQ_LIBRARIES = ${ZMQ_LIBRARIES}" )
else()
    message( FATAL_ERROR "ZMQ not found" )
endif()

# get all *.cpp files from src to build target
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*${SOURCE_EXT_PATTERN})
source_group("Source Files" FILES ${SOURCE_FILES})
message ("SOURCE FILES" = ${SOURCE_FILES})

# for grouping header files in IDE project generation
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*${HEADER_EXT_PATTERN})
source_group("Headers" FILES ${HEADER_FILES})

# set the executable output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# build target
add_executable( ${PROJECT_NAME} ${SOURCE_FILES} )
target_link_libraries( ${PROJECT_NAME} InmindEmotionDetector )
target_link_libraries( ${PROJECT_NAME} LandmarkDetector )
target_link_libraries( ${PROJECT_NAME} FaceAnalyser )
target_link_libraries( ${PROJECT_NAME} dlib )

target_link_libraries( ${PROJECT_NAME} ${FFMPEG_LIBRARIES} ${OpenCV_LIBS} ${ZMQ_LIBRARIES} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )
