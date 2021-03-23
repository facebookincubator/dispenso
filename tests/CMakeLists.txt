cmake_minimum_required(VERSION 3.12)

### TODO(bbudge): Set up testing with github actions
# Ideally these tests are all run in (Release, Debug) X (N/A, TSAN, ASAN, -fno-exceptions)
###

# Locate GTest
include(GoogleTest)

if (WIN32)
  add_subdirectory(../../thirdparty/googletest googletestDir)
  set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
else()
  find_package(GTest REQUIRED)
endif()

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_compile_features(${TESTNAME} PRIVATE cxx_std_14)
    target_link_libraries(${TESTNAME} ${GTEST_BOTH_LIBRARIES} dispenso)
    if(WIN32)
      target_link_libraries(${TESTNAME} Synchronization)
    else()
      target_link_libraries(${TESTNAME} pthread)
    endif()
    gtest_discover_tests(${TESTNAME}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB TEST_FILES CONFIGURE_DEPENDS *.cpp)

foreach(TEST_FILE ${TEST_FILES})
  set(TEST_NAME)
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  package_add_test(${TEST_NAME} ${TEST_FILE})
endforeach()
