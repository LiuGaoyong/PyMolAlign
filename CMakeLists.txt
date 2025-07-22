cmake_minimum_required(VERSION 3.17.2...3.26)

project(MolAlign
  VERSION 1.0
  LANGUAGES C Fortran
)

# Basic Library
add_subdirectory("fortran")
add_library(MolAlignLib STATIC ${srcs})
# two varible: f2pysrcs & srcs


# Start to Setup F2PY
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
) # F2PY headers

set(f2py_module_c "${f2py_module_name}module.c")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  COMMAND ${PYTHON_EXECUTABLE}  -m "numpy.f2py"
    "${f2pysrcs}"
    -m "molalign"
    --lower # Important
  DEPENDS fortran/molalign.pyf
)

python_add_library(${CMAKE_PROJECT_NAME} MODULE
                  "${f2py_module_name}module.c"
                  "${F2PY_INCLUDE_DIR}/fortranobject.c"
                  "${f2pysrcs}" WITH_SOABI
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
                           ${F2PY_INCLUDE_DIR}
                          )

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
                      Python::NumPy
                      MolAlignLib
)

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION pi)