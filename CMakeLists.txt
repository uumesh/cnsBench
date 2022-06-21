cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0074 NEW)

project(cnsbench)

enable_language(C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LIBP_DIR "/gpfs/alpine/scratch/umeshu/cfd144/forks/cmake-libp/libparanumal/install")

include(GNUInstallDirs)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

option(EXTERNAL_OCCA "Use an external OCCA build" OFF)
if(EXTERNAL_OCCA)
  message(STATUS "Using an external OCCA build.")
  find_package(OCCA REQUIRED)
  set(OCCA_TARGET OCCA::libocca)
else()
  message(FATAL_ERROR "Currently only external OCCA builds are enabled ")
endif()

add_compile_definitions(
    "USE_OCCA_MEM_BYTE_ALIGN=64"
    "dfloat=double"
    "dfloatString=\"double\""
    "dfloatFormat=\"%lf\""
    "MPI_DFLOAT=MPI_DOUBLE"
    "dlong=int"
    "dlongString=\"int\""
    "dlongFormat=\"%d\""
    "MPI_DLONG=MPI_INT"
    "hlong=int"
    "hlongString=\"int\""
    "hlongFormat=\"%d\""
    "MPI_HLONG=MPI_INT"
)

add_executable(cnsbench
  cnsBench.cpp
  src/cnsSettings.cpp
)

target_include_directories(cnsbench PUBLIC ${LIBP_DIR}/include ${LIBP_DIR}/include/ogs ${LIBP_DIR}/include/mesh ${LIBP_DIR}/../3rdParty/gslib/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(cnsbench PUBLIC ${LIBP_DIR}/lib64 ${LIBP_DIR}/../3rdParty/gslib/lib)
target_link_libraries(cnsbench
  PUBLIC
    timeStepper
    mesh
    ogs
    linAlg
    core
    OCCA::libocca
    MPI::MPI_CXX
)

# set_target_properties(cnsbench PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/cns.hpp)

# target_compile_definitions(cnsbench PUBLIC "LIBP_DIR=\"${CMAKE_INSTALL_PREFIX}\"")
target_compile_definitions(cnsbench PUBLIC "DCNS=\"${CMAKE_INSTALL_PREFIX}\"")

install(TARGETS cnsbench)
install(DIRECTORY okl/ DESTINATION okl)
