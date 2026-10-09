// How fast a prompt reads and a verify step runs, and where the time goes.
//
//   tools/op-profile.sh -m MODEL [llama flags...] -f PROMPT_FILE [-n TOKENS]
//
// Reads the prompt once to warm up, then once for its rate. Then it times 32
// steps of 4 tokens, the width of an MTP verify at --spec-draft-n-max 3, and
// reports them with and without the first two, which build the CUDA graphs.
// These runs set no eval callback: with one, the scheduler syncs after every
// split.
//
// OP_PROFILE_OPS=1 adds a pass of each under the callback, which asks for every
// node whose output is in host memory. Each CPU node then runs alone and its
// time is exact; the card, the copies and the waits are the rest of the wall
// time. Nodes run one at a time, so fusion is off and the shares are close to
// a normal run's, not equal.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

struct profile {
    clk::time_point asked;
    bool on = false;
    std::map<std::string, double> ms;     // op | name without layer number
    std::map<std::string, int>    count;
    double total = 0;
};

static std::string label(const ggml_tensor * t) {
    std::string name = t->name;
    // "ffn_moe_up-17" and "ffn_moe_up-17 (copy)" are one row
    auto dash = name.find('-');
    if (dash != std::string::npos) {
        name = name.substr(0, dash);
    }
    return std::string(ggml_op_desc(t)) + " | " + name;
}

static bool on_node(ggml_tensor * t, bool ask, void * data) {
    auto * p = (profile *) data;
    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
    if (!p->on || !host) {
        return false;
    }
    if (ask) {
        p->asked = clk::now();
        return true;
    }
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - p->asked).count();
    p->ms[label(t)] += ms;
    p->count[label(t)] += 1;
    p->total += ms;
    return true;
}

static void report(profile & p, int n, double plain, double wall, const char * what) {
    std::vector<std::pair<double, std::string>> rows;
    for (auto & [k, v] : p.ms) {
        rows.push_back({v, k});
    }
    std::sort(rows.rbegin(), rows.rend());
    printf("\n== %s: %d of them, %.0f ms without the callback (%.2f ms each), %.0f ms with it\n",
           what, n, plain, plain / n, wall);
    printf("CPU nodes %.0f ms (%.1f%%), the rest (card, copies, waits) %.0f ms (%.1f%%)\n\n",
           p.total, 100 * p.total / wall, wall - p.total, 100 * (wall - p.total) / wall);
    printf("%9s %6s %7s  %s\n", "ms", "%wall", "nodes", "op | tensor");
    for (size_t i = 0; i < rows.size() && i < 25; i++) {
        printf("%9.1f %5.1f%% %7d  %s\n", rows[i].first, 100 * rows[i].first / wall,
               p.count[rows[i].second], rows[i].second.c_str());
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    // The callback makes the scheduler sync after every split, so the plain timings run
    // without it. OP_PROFILE_OPS=1 adds the per-op tables, from a second pass with it.
    const bool ops = getenv("OP_PROFILE_OPS") != nullptr;
    profile p;
    if (ops) {
        params.cb_eval = on_node;
        params.cb_eval_user_data = &p;
    }
    params.warmup = false;

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    std::vector<llama_token> all = common_tokenize(ctx, params.prompt, true, true);
    const int n = std::min<int>((int) all.size(), params.n_predict > 0 ? params.n_predict : 2048);
    if (n < 2) {
        LOG_ERR("the prompt is %zu tokens, too short\n", all.size());
        return 1;
    }

    auto read = [&](int from) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const auto t0 = clk::now();
        for (int i = 0; i < n; i += params.n_batch) {
            const int len = std::min(params.n_batch, n - i);
            std::vector<llama_token> part(all.begin() + from + i, all.begin() + from + i + len);
            if (llama_decode(ctx, llama_batch_get_one(part.data(), len))) {
                LOG_ERR("decode failed\n");
                return -1.0;
            }
        }
        llama_synchronize(ctx);
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    read(0);                                   // warm: the first touch of each page
    const double plain = read(0);
    printf("\n== prompt: %d of them, %.0f ms (%.2f ms each)\n", n, plain, plain / n);
    if (ops) {
        p.on = true;
        const double wall = read(0);
        report(p, n, plain, wall, "prompt, with the callback");
        p.on = false;
    }

    // generate: verify-wide steps on top of the prompt just read. The first two after a
    // change of batch shape build the CUDA graphs, so "steady" leaves them out.
    const int steps = 32, width = 4, warm = 2;
    auto step = [&](int k) {
        std::vector<llama_token> part(all.begin() + n + k * width, all.begin() + n + (k + 1) * width);
        llama_batch b = llama_batch_init(width, 0, 1);
        for (int j = 0; j < width; j++) {
            b.token[j]     = part[j];
            b.pos[j]       = n + k * width + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = true;      // a verify reads every position
        }
        b.n_tokens = width;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc;
    };
    if ((int) all.size() < n + 2 * steps * width) {
        printf("the prompt file is too short for the generate steps\n");
    } else {
        double total = 0, steady = 0;
        for (int k = 0; k < steps; k++) {
            const auto t0 = clk::now();
            step(k);
            llama_synchronize(ctx);
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            total += ms;
            if (k >= warm) {
                steady += ms;
            }
        }
        printf("== verify steps of 4 tokens: %d of them, steady %.2f ms each (all %.2f ms each)\n",
               steps, steady / (steps - warm), total / steps);
        if (ops) {
            read(0);                           // a recurrent state cannot step back
            p = profile{};
            p.on = true;
            const auto t0 = clk::now();
            for (int k = 0; k < steps; k++) step(k);
            llama_synchronize(ctx);
            const double gwall = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            report(p, steps, total, gwall, "verify steps of 4 tokens, with the callback");
        }
    }
    llama_backend_free();
    return 0;
}
