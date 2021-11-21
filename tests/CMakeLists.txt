cmake_minimum_required(VERSION 3.14)

### TODO(bbudge): Set up testing with github actions
# Ideally these tests are all run in (Release, Debug) X (N/A, TSAN, ASAN, -fno-exceptions)
###

include(FetchContent)
FetchContent_Declare(
  GoogleTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

# For Windows, Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GoogleTest)
include(GoogleTest)

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_compile_features(${TESTNAME} PRIVATE cxx_std_14)
    target_compile_options(${TESTNAME} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W3 /WX>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wconversion -Wno-sign-conversion -Werror>
      )
    target_link_libraries(${TESTNAME} gmock_main dispenso)
    gtest_discover_tests(${TESTNAME}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB TEST_FILES CONFIGURE_DEPENDS "*test.cpp")
#TODO(elliotsegal, bbudge): Help add the shared_pool_test for CMake
LIST(REMOVE_ITEM TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shared_pool_test.cpp)

foreach(TEST_FILE ${TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} ${TEST_FILE})
endforeach()
