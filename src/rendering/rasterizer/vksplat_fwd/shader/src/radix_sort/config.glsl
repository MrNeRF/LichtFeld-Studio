#define SUBGROUP_SIZE 32

// #define KEY_BITS 64
// #define keyType uint64_t
#define KEY_BITS 32
#define keyType uint32_t

const int RADIX = 256;
#define WORKGROUP_SIZE 512
#define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
