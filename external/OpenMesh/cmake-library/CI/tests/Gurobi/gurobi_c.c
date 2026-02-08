#include <gurobi_c.h>

int main(int argc, char *argv[])
{
    int major, minor, technical;
    GRBversion(&major, &minor, &technical);
    return 0;
}
