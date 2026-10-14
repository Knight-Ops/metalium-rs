// A DRAM reader for the bandwidth comparison with metalium-rs: one core reads `num_pages` pages of an
// interleaved DRAM buffer into a window of its own L1, `pages_per_barrier` requests at a time, and stamps the
// tile's wall clock before the first request and after the last has landed.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t pages_per_barrier = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t l1_base = get_arg_val<uint32_t>(5);
    const uint32_t span_pages = get_arg_val<uint32_t>(6);
    const uint32_t result_addr = get_arg_val<uint32_t>(7);
    const uint32_t pinned = get_arg_val<uint32_t>(8);      // 1: read one DRAM bank directly, in order
    const uint32_t bank = get_arg_val<uint32_t>(9);
    const uint32_t bank_first = get_arg_val<uint32_t>(10);  // the first page's index within that bank

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, addr);

    const uint32_t t0 = get_timestamp_32b();
    uint32_t slot = 0;
    uint32_t i = 0;
    while (i < num_pages) {
        uint32_t n = num_pages - i;
        if (n > pages_per_barrier) {
            n = pages_per_barrier;
        }
        for (uint32_t j = 0; j < n; ++j) {
            if (pinned) {
                const uint64_t a = get_noc_addr_from_bank_id<true>(bank, addr + (bank_first + i + j) * page_size);
                noc_async_read(a, l1_base + slot * page_size, page_size);
            } else {
                noc_async_read_page(first_page + i + j, s, l1_base + slot * page_size);
            }
            slot = (slot + 1 == span_pages) ? 0 : slot + 1;
        }
        noc_async_read_barrier();
        i += n;
    }
    const uint32_t t1 = get_timestamp_32b();
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
    r[0] = t0;
    r[1] = t1;
}
