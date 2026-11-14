file(READ "${SPIRV}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])" "0x\\4\\3\\2\\1u," words "${bytes}")
file(WRITE "${OUTPUT}" "#pragma once\n#include <cstdint>\ninline constexpr std::uint32_t rindray_shader[] = {${words}};\n")
