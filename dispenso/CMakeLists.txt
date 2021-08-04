cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.h)

if (DISPENSO_SHARED_LIB)
  if (UNIX)
    add_compile_options("-fvisibility=hidden")
  endif()
  
  add_compile_definitions(DISPENSO_SHARED_LIB DISPENSO_LIB_EXPORT)
  add_library(dispenso SHARED ${SOURCES} ${HEADERS})
else()
  add_library(dispenso STATIC ${SOURCES} ${HEADERS})
endif()

target_compile_options(dispenso PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
)

target_include_directories(dispenso PUBLIC .. third-party/moodycamel "${PROJECT_BINARY_DIR}")

target_compile_features(dispenso PUBLIC cxx_std_14)
