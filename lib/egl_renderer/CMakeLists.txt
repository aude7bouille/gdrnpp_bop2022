cmake_minimum_required(VERSION 3.1)
# 3.8 gives us built-in CUDA support; 3.10 gives us OpenGL::EGL
project(CppEGLRenderer)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

#find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

set(EGL_INCLUDE_DIR /home/aude/Stage3A/MachineLearning/gdrnpp_bop2022/lib/egl_renderer/egl_include/EGL) # Modification
set(OPENGL_egl_LIBRARY /usr/lib/x86_64-linux-gnu/libEGL.so) # Modification
set(CMAKE_FIND_DEBUG_MODE TRUE) # Modification
find_library(EGL_LIBRARIES NAMES egl EGL REQUIRED)
set(CMAKE_FIND_DEBUG_MODE FALSE) # Modification

include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})

cuda_add_library(CppEGLRenderer MODULE  cpp/egl_renderer.cpp)
target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL)
set_target_properties(CppEGLRenderer PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_executable(query_devices cpp/query_devices.cpp)
add_executable(test_device cpp/test_device.cpp)

target_link_libraries(query_devices GL EGL dl pthread)
target_link_libraries(test_device GL EGL dl pthread)


#if(FALSE)

# could not solve the problem with OpenGL, despite having tried several things
# (like adding set(OpenGL_GL_PREFERENCE GLVND),
# set(OPENGL_gl_LIBRARY /usr/lib/x86_64-linux-gnu/libGL.so)
# and set(OPENGL_egl_LIBRARY /usr/lib/x86_64-linux-gnu/libEGL.so)
# in the CMakeLists file)


cmake_minimum_required(VERSION 3.1)
# 3.8 gives us built-in CUDA support; 3.10 gives us OpenGL::EGL
project(CppEGLRenderer)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})


#find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)

#set(OpenGL_GL_PREFERENCE GLVND) # Modification
#set(OPENGL_gl_LIBRARY /usr/lib/x86_64-linux-gnu/libGL.so) # Modification
#set(OPENGL_egl_LIBRARY /usr/lib/x86_64-linux-gnu/libEGL.so) # Modification


find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
#include_directories(${/usr/lib/x86_64-linux-gnu}) # Modification

# Search for EGL; nvidia drivers ship the library but not headers, so we redistribute those
# /usr/include (Ubuntu>=18.04) or ./egl_include
find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/egl_include REQUIRED)
message(STATUS "EGL_INCLUDE_DIR: ${EGL_INCLUDE_DIR}")
find_library(EGL_LIBRARIES NAMES egl EGL REQUIRED)


include_directories(SYSTEM ${EGL_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})

cuda_add_library(CppEGLRenderer MODULE  cpp/egl_renderer.cpp)
# Equivalent to pass flags -lGL, -lGLU and -lglut
target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL)
set_target_properties(CppEGLRenderer PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")


set(EGL_HEADER_FILES ${EGL_INCLUDE_DIR}/EGL/egl.h ${EGL_INCLUDE_DIR}/EGL/eglext.h)  # TODO: remove for ubuntu >= 18.04


add_executable(query_devices cpp/query_devices.cpp ${EGL_HEADER_FILES})
target_include_directories(query_devices PRIVATE ${EGL_INCLUDE_DIR})  # TODO: remove for ubuntu >= 18.04

add_executable(test_device cpp/test_device.cpp ${EGL_HEADER_FILES})
target_include_directories(test_device PRIVATE ${EGL_INCLUDE_DIR})  # TODO: remove for ubuntu >= 18.04

target_link_libraries(query_devices GL EGL dl pthread)
target_link_libraries(test_device GL EGL dl pthread)
#endif