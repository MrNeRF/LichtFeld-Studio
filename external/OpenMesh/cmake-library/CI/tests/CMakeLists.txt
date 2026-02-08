
file(GLOB tests 
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*")

foreach(test ${tests})
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${test}")
        message(STATUS "add_subdirectory(${test})")
        add_subdirectory("${test}")
    endif()
endforeach()


#[[==
maybe run configure/build as test?
cf https://cliutils.gitlab.io/modern-cmake/chapters/testing.html

add_test(
  NAME
    ExampleCMakeBuild
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${My_SOURCE_DIR}/examples/simple"
                              "${CMAKE_CURRENT_BINARY_DIR}/simple"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
==]]
