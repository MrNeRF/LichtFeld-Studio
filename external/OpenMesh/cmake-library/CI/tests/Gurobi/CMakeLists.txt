find_package(Gurobi REQUIRED)

add_executable(gurobi_c "gurobi_c.c")
target_link_libraries(gurobi_c Gurobi::GurobiC)
add_test(NAME gurobi_c COMMAND gurobi_c)

add_executable(gurobi_cxx "gurobi_cxx.cc")
target_link_libraries(gurobi_cxx Gurobi::GurobiCXX)
add_test(NAME gurobi_cxx COMMAND gurobi_cxx)


