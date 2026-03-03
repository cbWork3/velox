#pragma once

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef DEBUG

#define print_bool(addr,size)
#define print_u8(addr,size)
#define print_hex(addr,size)
#define print_int(addr,size)
#define print_u32_hex(addr,size)
#define print_u64(addr,size)
#define print_u64_hex(addr,size)
#define debug_cout(x)

#else

#define print_bool(addr,size)
#define print_u8(addr,size)
#define print_hex(addr,size)
#define print_int(addr,size)
#define print_u32_hex(addr,size)
#define print_u64(addr,size)
#define print_u64_hex(addr,size)
#define debug_cout(x)

#endif

#define _KB (1024ULL)
#define _MB (1024ULL * 1024ULL)
#define _GB (1024ULL * 1024ULL * 1024ULL)

const size_t BLOOM_GROUP_SIZE=256;

const size_t HT_GROUP_SIZE=96;