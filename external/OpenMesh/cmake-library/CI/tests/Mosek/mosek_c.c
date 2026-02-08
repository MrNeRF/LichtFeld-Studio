#include <mosek.h>

int main(int argc, char *argv[])
{
    int major, minor, revision;
    MSK_getversion(&major, &minor, &revision);
    printf("Mosek version: %d.%d.%d\n", major, minor, revision);
    return 0;
}
