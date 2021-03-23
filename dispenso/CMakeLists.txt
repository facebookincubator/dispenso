cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)

add_library(dispenso ${SOURCES} ${HEADERS})

target_include_directories(dispenso PUBLIC .. third-party/moodycamel)

target_compile_features(dispenso PUBLIC cxx_std_14)
