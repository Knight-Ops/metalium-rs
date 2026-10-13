// DRAM -> L1 read bandwidth on tt-metal, the counterpart of metalium-rs's `gddr_aggregate` Read mix:
// N cores each read their share of one interleaved DRAM buffer, page by page, `--batch` requests per barrier,
// timed on the tile's wall clock from inside the kernel. Every iteration's bytes are checked.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

static uint64_t arg_u64(int argc, char** argv, const char* name, uint64_t def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (!strcmp(argv[i], name)) return strtoull(argv[i + 1], nullptr, 0);
    }
    return def;
}

int main(int argc, char** argv) {
    const uint32_t page = arg_u64(argc, argv, "--page", 4096);
    const uint32_t batch = arg_u64(argc, argv, "--batch", 8);
    const uint32_t noc_sel = arg_u64(argc, argv, "--noc", 0);
    const bool pinned = arg_u64(argc, argv, "--pinned", 0) != 0;
    const uint32_t iters = arg_u64(argc, argv, "--iters", 7);
    const uint32_t pages_per_core = arg_u64(argc, argv, "--pages", (1u << 20) / page);
    const std::string kernel_path = std::getenv("READER_KERNEL") ? std::getenv("READER_KERNEL") : "reader.cpp";
    std::vector<uint32_t> counts;
    {
        std::string s = std::getenv("CORES") ? std::getenv("CORES") : "1,2,4,8,16,32,64,120";
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            counts.push_back(std::stoul(s.substr(p, q == std::string::npos ? q : q - p)));
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    bool ok = true;
    try {
        auto mesh = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh->mesh_command_queue();
        auto* dev = mesh->get_devices()[0];
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t max_cores = grid.x * grid.y;
        const uint32_t nb = dev->num_dram_channels();
        const uint32_t l1_base = mesh->allocator()->get_base_allocator_addr(HalMemType::L1);
        const uint32_t span_pages = std::max<uint32_t>(1, (256u * 1024) / page);
        const uint32_t result_addr = l1_base + span_pages * page;
        const double mhz = 1350.0;
        printf("INFO grid %ux%u (%u cores), page %u B, %u pages per core, batch %u, noc %u, l1_base 0x%x\n", grid.x,
               grid.y, max_cores, page, pages_per_core, batch, noc_sel, l1_base);

        for (uint32_t n : counts) {
            if (n > max_cores) {
                printf("SKIP cores %u: only %u available\n", n, max_cores);
                continue;
            }
            // Pinned: core i owns bank (i % nb), pages [(i / nb) * P, +P) of it; the buffer holds that many per bank.
            const uint64_t total_pages =
                pinned ? uint64_t(nb) * ((n + nb - 1) / nb) * pages_per_core : uint64_t(n) * pages_per_core;
            const uint64_t bytes_per_core = uint64_t(pages_per_core) * page;
            const uint64_t buf_bytes = total_pages * page;
            distributed::DeviceLocalBufferConfig dcfg{.page_size = page, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig bcfg{.size = buf_bytes};
            auto buf = distributed::MeshBuffer::create(bcfg, dcfg, mesh.get());
            // Page k holds the word k in every position: a read that lands the wrong page, or none, shows.
            std::vector<uint32_t> host(buf_bytes / 4);
            for (uint64_t k = 0; k < total_pages; ++k)
                std::fill_n(host.begin() + k * (page / 4), page / 4, uint32_t(k + 1));
            distributed::EnqueueWriteMeshBuffer(cq, buf, host, false);
            distributed::Finish(cq);

            CoreRangeSet crs = num_cores_to_corerangeset(n, grid, true);
            std::vector<CoreCoord> cores = corerange_to_cores(crs, n, true);
            Program program = CreateProgram();
            std::vector<uint32_t> cta;
            TensorAccessorArgs(*buf->get_backing_buffer()).append_to(cta);
            auto kid = CreateKernel(
                program, kernel_path, crs,
                DataMovementConfig{
                    .processor = noc_sel == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
                    .noc = noc_sel == 0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                    .compile_args = cta});
            for (uint32_t i = 0; i < n; ++i) {
                SetRuntimeArgs(program, kid, cores[i],
                               std::vector<uint32_t>{(uint32_t)buf->address(), i * pages_per_core, pages_per_core,
                                                     batch, page, l1_base, span_pages, result_addr,
                                                     pinned ? 1u : 0u, i % nb, (i / nb) * pages_per_core});
            }
            distributed::MeshWorkload wl;
            wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));

            std::vector<double> gbps, per_core;
            for (uint32_t it = 0; it < iters + 1; ++it) {
                // The destination windows poisoned, so a read that did not land cannot pass.
                std::vector<uint32_t> poison(span_pages * page / 4 + 2, 0xA5A5A5A5u);
                for (auto& c : cores) tt::tt_metal::detail::WriteToDeviceL1(dev, c, l1_base, poison);
                distributed::EnqueueMeshWorkload(cq, wl, false);
                distributed::Finish(cq);
                int64_t first_start = INT64_MAX, last_end = INT64_MIN;
                uint32_t ref = 0;
                std::vector<double> core_rates;
                for (uint32_t i = 0; i < n; ++i) {
                    std::vector<uint32_t> r;
                    tt::tt_metal::detail::ReadFromDeviceL1(dev, cores[i], result_addr, 8, r);
                    if (i == 0) ref = r[0];
                    const int64_t s = int32_t(r[0] - ref), e = int32_t(r[1] - ref);
                    first_start = std::min(first_start, s);
                    last_end = std::max(last_end, e);
                    core_rates.push_back(double(bytes_per_core) / (double(uint32_t(r[1] - r[0])) / (mhz * 1e6)) / 1e9);
                }
                // Landed: the last window of two cores, page by page.
                for (uint32_t ci : {0u, n - 1}) {
                    std::vector<uint32_t> w;
                    tt::tt_metal::detail::ReadFromDeviceL1(dev, cores[ci], l1_base, span_pages * page, w);
                    for (uint32_t slot = 0; slot < span_pages; ++slot) {
                        const uint32_t last_i = ((pages_per_core - 1 - slot) / span_pages) * span_pages + slot;
                        const uint32_t want = pinned ? ((ci / nb) * pages_per_core + last_i) * nb + (ci % nb) + 1
                                                     : ci * pages_per_core + last_i + 1;
                        if (w[slot * (page / 4)] != want || w[slot * (page / 4) + page / 4 - 1] != want) {
                            ok = false;
                            printf("MISMATCH core %u slot %u: got %u want %u\n", ci, slot, w[slot * (page / 4)], want);
                            break;
                        }
                    }
                }
                if (it == 0) continue;  // warm-up (JIT build, first touch)
                const double together_s = double(last_end - first_start) / (mhz * 1e6);
                gbps.push_back(double(bytes_per_core) * n / together_s / 1e9);
                std::sort(core_rates.begin(), core_rates.end());
                per_core.push_back(core_rates[core_rates.size() / 2]);
            }
            std::sort(gbps.begin(), gbps.end());
            std::sort(per_core.begin(), per_core.end());
            auto at = [&](std::vector<double>& v, double q) { return v[size_t((v.size() - 1) * q + 0.5)]; };
            printf("RESULT pinned=%d page=%u cores=%u batch=%u noc=%u gbps_med=%.2f p10=%.2f p90=%.2f per_core_med=%.2f pct512=%.1f\n",
                   int(pinned), page, n, batch, noc_sel, at(gbps, 0.5), at(gbps, 0.1), at(gbps, 0.9), at(per_core, 0.5),
                   100.0 * at(gbps, 0.5) / 512.0);
            fflush(stdout);
        }
        mesh->close();
    } catch (const std::exception& e) {
        printf("EXCEPTION %s\n", e.what());
        ok = false;
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
