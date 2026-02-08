#include <gurobi_c++.h>
#include <iostream>

int main(int argc, char *argv[])
{
    try {
        GRBEnv env;
        GRBModel model {env};
    } catch (GRBException &e) {
        std::cerr << "Gurobi exception: " << e.getMessage() << std::endl;
    }
    return 0;
}
