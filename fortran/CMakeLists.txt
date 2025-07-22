set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(APPEND srcs
    "${dir}/stdio.f90"
    "${dir}/kinds.f90"
    "${dir}/flags.f90"
    "${dir}/bounds.f90"
    "${dir}/strutils.f90"
    "${dir}/jacobi.f90"
    "${dir}/permutation.f90"
    "${dir}/lap_sparse.f90"
    "${dir}/rnglib.f90"
    "${dir}/random.f90"
    "${dir}/sorting.f90"
    "${dir}/biasing.f90"
    "${dir}/printing.f90"
    "${dir}/chemdata.f90"
    "${dir}/chemutils.f90"
    "${dir}/assorting.f90"
    "${dir}/translation.f90"
    "${dir}/rotation.f90"
    "${dir}/alignment.f90"
    "${dir}/assignment.f90"
    "${dir}/library.f90"
)
set(srcs ${srcs} PARENT_SCOPE)


list(APPEND f2pysrcs
    "${dir}/kinds.f90"
    "${dir}/flags.f90"
    "${dir}/bounds.f90"
    "${dir}/biasing.f90"
    "${dir}/library.f90"
)
set(f2pysrcs ${f2pysrcs} PARENT_SCOPE)