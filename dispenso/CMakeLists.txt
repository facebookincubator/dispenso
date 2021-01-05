cmake_minimum_required(VERSION 3.12)

############# Libraries

file(GLOB SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/dispenso/*.cpp ${PROJECT_SOURCE_DIR}/dispenso/detail/*.cpp)
file(GLOB HDR_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/dispenso/*.h ${PROJECT_SOURCE_DIR}/dispenso/detail/*.h ${PROJECT_SOURCE_DIR}/dispenso/third-party/moodycamel/*.h)

add_library(
        dispenso
        ${HDR_FILES}
        ${SRC_FILES}
)
