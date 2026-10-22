# A2/A3 compiler sources are built with the existing USE_ASCEND gate. They
# transform IR or emit source text and do not include CANN headers or link CANN
# libraries. The parent Ascend CMake file owns the shared runtime/stub choice.
# catlass, shmem, and pto-isa headers belong to the later Bisheng JIT compile,
# so their availability must not gate this native, source-only build.
list(APPEND TILE_LANG_SRCS
  src/ascend/a2a3/transform/ascend_host.cc
)
