find_package(GMM REQUIRED)

add_executable(test_gmm "test_gmm.cc")
target_link_libraries(test_gmm PRIVATE GMM::GMM)
add_test(NAME test_gmm COMMAND test_gmm)


