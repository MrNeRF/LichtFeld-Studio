#include <fusion.h>
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "Mosek FusionCXX version "
              << mosek::fusion::Model::getVersion()
              << std::endl;
    return 0;
}
