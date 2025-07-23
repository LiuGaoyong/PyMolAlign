cmake_minimum_required(VERSION 3.17.2...3.26)

project(
  _molalign_qcuaeh
  VERSION 1.0
  DESCRIPTION "Python module for molecular alignment"
  HOMEPAGE_URL "https://github.com/qcuaeh/molalignlib"
  LANGUAGES C Fortran
)


add_subdirectory("fortran")
add_library(MolAlignLib STATIC ${srcs})
set(fortran_src_file ${f2pysrcs})


# Python, Numpy & F2PY headers
find_package(
  Python REQUIRED
  COMPONENTS Interpreter
  Development.Module NumPy
)
include(UseF2Py)


f2py_object_library(f2py_object OBJECT)
f2py_generate_module(
  ${CMAKE_PROJECT_NAME}
  ${fortran_src_file}
  OUTPUT_VARIABLE f2py_files
)
python_add_library(
  ${CMAKE_PROJECT_NAME} MODULE
  ${f2py_files}
  ${fortran_src_file} WITH_SOABI)
target_link_libraries(
  ${CMAKE_PROJECT_NAME} PRIVATE
  f2py_object MolAlignLib)
install(
  TARGETS ${CMAKE_PROJECT_NAME}
  DESTINATION pymolalign)