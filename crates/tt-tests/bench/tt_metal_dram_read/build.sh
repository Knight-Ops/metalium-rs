#!/bin/bash
# Build dram_read_bw against tt-metal v077's installed libraries (flags from `ninja -t commands`, mem_bench).
set -e
R=/home/carl/projects/tt-metal-v077
B=$R/build_Release
C=/home/carl/projects/tt-metal/.cpmcache
cd $(dirname "$0")
/usr/bin/clang++-20 -DFMT_HEADER_ONLY=1 -DNTEST -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DTRACY_ENABLE -DTRACY_IMPORTS \
  -DTRACY_TIMER_FALLBACK -DTT_ENABLE_LIGHT_METAL_TRACE=1 -DTT_UMD_BUILD_SIMULATION -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 \
  -I$R/tt_metal/api -I$R/tt_metal/api/tt-metalium -I$B/tt_metal/api -I$R -I$R/tt_metal -I$R/tt_metal/common -I$R/tests \
  -I$R/tt_metal/hostdevcommon/api -I$R/tt_stl/. -I$R/tt_metal/hw/inc -I$R/tt_metal/hw \
  -isystem $R/tt_metal/third_party/umd/device/api \
  -isystem $C/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include \
  -isystem $C/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include \
  -isystem $R/tt_metal/third_party/tracy/public -isystem $B/generated/profiler \
  -isystem $C/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include \
  -isystem $C/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0 \
  -isystem $C/tt-logger/d2339ce68562cae34cd95f3fece7fd94eb0529b7/include \
  -isystem $C/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include \
  -O2 -std=c++20 -fPIE -fsized-deallocation -march=x86-64-v3 -Wno-deprecated-declarations -Wno-int-to-pointer-cast \
  dram_read_bw.cpp -o dram_read_bw -fuse-ld=lld -rdynamic \
  -Wl,-rpath,$B/tt_metal:$B/lib:$B/tt_stl:$B/tt_metal/third_party/umd/lib \
  $B/tt_metal/libtt_metal.so -lnuma $B/lib/libtracy.so.0.13.3 -ldl $B/tt_stl/libtt_stl.so $B/_deps/spdlog-build/libspdlog.a \
  $B/tt_metal/third_party/umd/lib/libtt-umd.so.0.77.0
echo built
