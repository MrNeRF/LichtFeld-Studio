find_package(Mosek REQUIRED)

add_executable(mosek_c "mosek_c.c")
target_link_libraries(mosek_c Mosek::MosekC)
add_test(NAME mosek_c COMMAND mosek_c)

add_executable(mosek_cxx "mosek_cxx.cc")
target_link_libraries(mosek_cxx Mosek::FusionCXX)
add_test(NAME mosek_cxx COMMAND mosek_cxx)


