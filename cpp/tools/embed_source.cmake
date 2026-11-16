# Embed a text file in the library as a C string:
#
#   cmake -DIN=<file> -DOUT=<file.c> -DSYMBOL=<name> -P embed_source.cmake
#
# writes `const char <name>[]` (NUL-terminated) and `const size_t <name>_len` (without the NUL).
# Used for cpp/kernels/bcp.cl, so the OpenCL kernel source a library compiles at run time is
# always the one it was built with. Bytes, not a string literal: no escaping and no length limit.
file(READ "${IN}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
get_filename_component(name "${IN}" NAME)
file(WRITE "${OUT}"
"/* GENERATED from ${name} by cpp/tools/embed_source.cmake -- DO NOT EDIT. */
#include <stddef.h>
const char ${SYMBOL}[] = {${bytes}0x00};
const size_t ${SYMBOL}_len = sizeof(${SYMBOL}) - 1;
")
