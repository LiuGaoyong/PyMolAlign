cmake_minimum_required(VERSION 3.17.2...3.26)

project(
  _molalign_qcuaeh
  VERSION 1.0
  DESCRIPTION "Python module for molecular alignment"
  HOMEPAGE_URL "https://github.com/qcuaeh/molalignlib"
  LANGUAGES C Fortran
)

########################################################################
################      External Fortran Library        ##################
########################################################################
set(srcs)
set(flib "molalign_fortran")
add_subdirectory("fortran")
add_library(${flib} STATIC ${srcs})

########################################################################
################        Numpy F2PY Library          ####################
########################################################################
# Python, Numpy & F2PY headers, fortranobject dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
include(UseF2Py)
# Create the F2Py `numpyobject` library.
f2py_object_library(f2py_object OBJECT)

set(fortran_src_file ${f2pysrcs})
set(py_module_name "_molalignlib")
f2py_generate_module("${py_module_name}" "${fortran_src_file}" OUTPUT_VARIABLE f_files)
MESSAGE(STATUS "This var key = ${f_files}.")
python_add_library("${py_module_name}" MODULE "${f_files}" "${fortran_src_file}" WITH_SOABI)
# target_link_libraries("${py_module_name}" PRIVATE f2py_object "${fortran_src_file}" ${flib})
# install(TARGETS ${py_module_name} DESTINATION pymolalign)