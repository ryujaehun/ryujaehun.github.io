---
title: "5. The decode path that produces one token — why separate kernels were made"
date: "2026-09-17"
draft: false
slug: "code-series-jmaczan--tiny-vllm-05"
categories:
  - code-series
series_id: "jmaczan--tiny-vllm"
chapter_order: 5
chapter_count: 7
repository: "https://github.com/jmaczan/tiny-vllm"
pinned_commit: "e25bf1994efa90bc98b721ba7c527402f86fbeaf"
---
Once prefill has streamed the whole prompt through in one pass, everything after that has to produce tokens **one at a time** based on the tokens produced so far. That step is decode. As we saw in part 1, prefill and decode have different operation shapes — the former is matrix × matrix, the latter vector × matrix — so this repo makes separate kernels. They're the variants with `Decode` in the name, like `embeddingGatherKernelDecode`, `ropeKernelDecode`, `softmaxKernelDecode`, and one more `pagedAttentionKernel` with no prefill counterpart. This part lays those kernels next to their prefill counterparts and reads **what differs and why the shape is different**, and how a new token's K/V gets appended to the cache.

> All citations in this series are against commit [`e25bf19`](https://github.com/jmaczan/tiny-vllm/tree/e25bf1994efa90bc98b721ba7c527402f86fbeaf). The author has no NVIDIA GPU, so nothing was built or run; all descriptions below come from reading the source.

## The decode loop: where it starts

Decode doesn't run inside the `prefill` function but inside the `while (true)` loop in `main`([`src/main.cpp:720-1039`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L720-L1039)). Each iteration starts by rebuilding the slot state. `active_slots` and `active_tokens` are cleared([`src/main.cpp:722-723`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L722-L723)), if there's an empty slot `prefill` is called again with the next prompt in the queue([`src/main.cpp:724-737`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L724-L737)), and only the occupied slots are collected and uploaded to the GPU. That slot·queue lifetime is part 7's subject. The key thing this part follows is the length information built and uploaded at the head of the loop.

```cpp
        cudaMemcpy(gpu_last_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_active_slots, active_slots.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> seq_lens(num_active_slots);
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            seq_lens[slot] = current_prompt_len[active_slot] + 1;
        }
        cudaMemcpy(gpu_seq_lens, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
```
— [`src/main.cpp:749-757`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L749-L757)

Why the length is uploaded as `current_prompt_len + 1` comes up later. Inside the layer loop the current token's K/V is written to the cache before the attention call([`src/main.cpp:851-873`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L851-L873)), and the attention kernel computes the number of blocks to iterate over from `gpu_seq_lens`([`src/kernels.cu:470-481`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L470-L481)), so the length has to contain **this token included**.

## The decode-family kernels versus their prefill counterparts

Decode calls four kernels that prefill doesn't. Three are variants of prefill counterparts, and one is a dedicated attention with no prefill equivalent. The implementations are separated in the decode section of `src/kernels.cu`([`src/kernels.cu:346-523`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L346-L523)) from the prefill section([`src/kernels.cu:26-344`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L26-L344)), and the declarations are gathered in the decode section of the kernel header `src/kernels.cuh`(the header we read together in part 3).

```cpp
// decode
void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens);
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim);
void softmaxDecode(__nv_bfloat16 *input, int seq_len);

// pagedattn
void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output);
```
— [`src/kernels.cuh:22-28`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cuh#L22-L28)

Where the two families split is shown in one table.

<!-- visual: prefill-decode-contrast supports: [prefill-decode-contrast] -->
| Aspect | prefill | decode | Evidence |
| --- | --- | --- | --- |
| Unit of work | the prompt's `prompt_len` tokens at once | 1 token per active slot | [`src/main.cpp:150-158`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L150-L158), [`src/main.cpp:759`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L759) |
| Embedding kernel | `embeddingGatherKernel`, `blockIdx.x` = token position | `embeddingGatherKernelDecode`, `blockIdx.x` = slot | [`src/kernels.cu:32-40`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L32-L40), [`src/kernels.cu:347-356`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L347-L356) |
| RoPE | `ropeKernel_llama3`, rotates every position with a prebuilt table | `ropeKernelDecode`, recomputes just the one current position per call | [`src/kernels.cu:173-200`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L173-L200), [`src/kernels.cu:371-384`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L371-L384) |
| Attention preprocessing | `causalMask` + `softmax` calls | online softmax inside `pagedAttentionKernel` | [`src/main.cpp:329-331`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L329-L331), [`src/kernels.cu:509-519`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L509-L519) |
| K/V write | copies a `BLOCK_SIZE`-token batch at once | copies 1 token, allocates a new block only at a block boundary | [`src/main.cpp:251-288`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L251-L288), [`src/main.cpp:851-873`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L851-L873) |
| cuBLAS `n` | `prompt_len` | `num_active_slots` | [`src/main.cpp:182`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L182), [`src/main.cpp:769`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L769) |
| RMSNorm·SwiGLU·residual | same kernel, only the token-count argument differs | same kernel, only the token-count argument differs | [`src/kernels.cu:55-81`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L55-L81), [`src/kernels.cu:311-329`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L311-L329), [`src/kernels.cu:331-338`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L331-L338) |

These two axes of the table are the answer to this part. **The decode-family kernels assume sequence length 1 and simplify the indexing accordingly.** `embeddingGatherKernelDecode` uses `blockIdx.x` as the slot rather than the token([`src/kernels.cu:349`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L349)), and `softmaxKernelDecode` assumes the `MAX_SEQ_LEN`-stride layout([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413)). And `ropeKernelDecode` rotates only **the one current position**, not every position. Each kernel is examined below.

## Embedding: `blockIdx.x` is the slot

Embedding is the first step that turns a token ID into a 2048-dimensional vector. Prefill's `embeddingGatherKernel` has one block handle one token. Launched as `<<<num_input_tokens, 1024>>>`, `blockIdx.x` is the token position, and `workIndex = blockIdx.x * 2048 + threadIdx.x` determines the output row([`src/kernels.cu:32-45`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L32-L45)). Decode's `embeddingGatherKernelDecode` has the same shape but a different meaning. The input isn't a token-ID array but the per-slot last-token array `gpu_last_tokens`, and `blockIdx.x` points at a slot.

```cpp
__global__ void embeddingGatherKernelDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens)
{
    int input_token = gpu_last_tokens[blockIdx.x];
    int workIndex = blockIdx.x * 2048 + threadIdx.x;
    if (workIndex < num_tokens * 2048)
    {
        output[workIndex] = embed_tokens[input_token * 2048 + threadIdx.x];
        output[workIndex + 1024] = embed_tokens[input_token * 2048 + threadIdx.x + 1024];
    }
}
```
— [`src/kernels.cu:347-356`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L347-L356)

`input_token = gpu_last_tokens[blockIdx.x]` — `blockIdx.x` is the slot index. The output is stacked in slot order too, so the `blockIdx.x` in `workIndex` is also the slot. The call site is `embeddingGatherDecode(gpu_last_tokens, num_active_slots, hidden_state, ...)`([`src/main.cpp:759`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L759)), where `num_active_slots` becomes the number of blocks([`src/kernels.cu:361`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L361)). It keeps the "one token per block" structure and just swaps the token axis for the slot axis.

## RoPE: one current position, recomputed on every call

RoPE (rotary position embedding) is a technique that bakes position information into a vector as a rotation. Since a vector rotates by a different angle at a different position, the rotation angle is a function of the position. The [visual explainer](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding) is the material the repo README gives as background.

Decode's RoPE diverges from prefill's at two points. First, it rotates only one position at a time. `ropeKernelDecode` is launched as a single block([`src/kernels.cu:397`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L397)), and each thread rotates the pair `2*threadIdx.x` and `2*threadIdx.x+1` together. The rotation angle is computed only for the single `position_in_sequence` passed as an argument([`src/kernels.cu:378`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L378)). Per slot, the call site

```cpp
                ropeDecode(&q_proj[slot * EMBEDDING_LENGTH], current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(k_proj_batched_buffer + slot * KV_DIM, current_prompt_len[active_slot], KV_DIM);
```
— [`src/main.cpp:847-848`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L847-L848)

The position argument is the index of the token this slot is about to produce, i.e., `current_prompt_len`. Q is 2048-dimensional and K is 512-dimensional, so `proj_dim` differs.

Second, decode recomputes the angle every time. This is the point to watch most closely in this part.

```cpp
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = position_in_sequence * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1];
        input[2 * threadIdx.x] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
```
— [`src/kernels.cu:375-382`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L375-L382)

`500000.0` and `32` are hardcoded constants([`src/kernels.cu:376-377`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L376-L377)). The comment right above leaves "precomputing thetas, angles and sin/cos and reusing them across all kernel invocations" as a TODO([`src/kernels.cu:375`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L375)). This recomputation is where the author left off.

By contrast, prefill's `ropeKernel_llama3` takes `d_cos_table`/`d_sin_table` as arguments and only reads `cos_table[token_idx * head_dim + pair_idx * 2]`([`src/kernels.cu:173-200`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L173-L200)). The table is built at the start of `main` by `init_rope_frequencies(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f, 8192)` and uploaded to the GPU([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572); the build and H2D copy are at [`src/kernels.cu:96-152`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L96-L152)).

There's one difference easy to miss here. The table path applies Llama 3's frequency scaling. `init_rope_frequencies` starts from the base `inv_freq[i] = 1/pow(rope_theta, 2i/head_dim)`, and when a wavelength is longer than `original_max_len / low_freq_factor`(=8192) it divides the frequency by `factor`(=32), interpolating in the band down to `high_freq_factor`(=4).

```cpp
    float low_freq_wavelen = (float)original_max_len / low_freq_factor;
    float high_freq_wavelen = (float)original_max_len / high_freq_factor;

    std::vector<float> inv_freq_llama = inv_freq;

    for (int i = 0; i < half_dim; i++)
    {
        float wavelen = 2.0f * M_PI / inv_freq[i];

        if (wavelen > low_freq_wavelen)
        {
            inv_freq_llama[i] = inv_freq[i] / factor;
        }
        else if (wavelen >= high_freq_wavelen)
        {
            float smooth = ((float)original_max_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            inv_freq_llama[i] = (1.0f - smooth) * (inv_freq[i] / factor) + smooth * inv_freq[i];
        }
    }
```
— [`src/kernels.cu:106-124`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L106-L124)

Decode's recomputation skips this scaling and uses plain `theta = 1/pow(500000, 2i/64)`. So in the low-frequency dimensions (the high-index dimensions), the two paths' rotation angles aren't numerically equal. The prefill table carries the Llama 3-scaled angles, while decode recomputes the unscaled `theta` on every call, and whether the two paths produce the same rotation has not been verified by running.

## Gathering K/V into a batched buffer and scattering into blocks

Decode's K/V projection groups slots as rows and handles them in one cuBLAS call. The K projection is called with `n = num_active_slots` and writes into `k_proj_batched_buffer`([`src/main.cpp:802-821`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L802-L821)), and V writes into `v_proj_batched_buffer` with the same shape([`src/main.cpp:824-842`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L824-L842)). Both buffers are allocated separately for decode, sized `BATCH_SIZE × KV_DIM`([`src/main.cpp:683-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L683-L693)). The transpose trick itself is the same skeleton we saw in part 4 — the only change is that `n` shrinks from `prompt_len` to the number of slots. After the per-slot RoPE([`src/main.cpp:844-849`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L844-L849)), the new token's K/V is copied into the appropriate block of the paged KV cache. The paged KV cache splits KV into small blocks of 16 tokens and uses a block table to track "which batch of which sequence is in which block" (the 2GB `cudaMalloc` we saw in part 1 is split into such blocks). Block layout and indexing are part 6's subject.

The point where the copy differs from prefill is that it handles **one token at a time**. For each slot, the new token's position is taken as `current_prompt_len` and split into a logical block and an in-block offset.

```cpp
                int seq_len = current_prompt_len[active_slot]; // + generated tokens?
                int logical_block_idx = seq_len / BLOCK_SIZE;
                int token_in_block_idx = seq_len % BLOCK_SIZE;
                int block = block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx];
                if (token_in_block_idx == 0)
                {
                    int physical_block_idx = free_blocks.back();
                    free_blocks.pop_back();
                    block = physical_block_idx;
                    block_table[active_slot * N_LAYERS * MAX_BLOCKS_PER_SEQ + layer * MAX_BLOCKS_PER_SEQ + logical_block_idx] = block;
                }
                __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *k_proj_ptr = k_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(k_cache_ptr, k_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)((char *)kv_cache + block * BLOCK_BYTES + V_OFFSET + token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16));
                __nv_bfloat16 *v_proj_ptr = v_proj_batched_buffer + slot * KV_DIM;
                cudaMemcpy(v_cache_ptr, v_proj_ptr, KV_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
```
— [`src/main.cpp:855-872`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L855-L872)

A new block is allocated only when the offset is 0, i.e., when this is the first token of a new block([`src/main.cpp:859-865`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L859-L865)). Otherwise the already-assigned block is reused. K is written via a D2D `cudaMemcpy` to block start + `token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16)`, and V to that position plus `V_OFFSET`. `V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16)`([`src/main.cpp:33`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L33)), and one block occupies `BLOCK_BYTES` including both K and V([`src/main.cpp:34`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L34)).

Prefill does the same thing in `BLOCK_SIZE`-token batches. For each token span it allocates a block and copies the whole batch at once([`src/main.cpp:251-288`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L251-L288)). Decode is that batch reduced to one token. After the write, the whole `block_table` is synchronized to `block_table_gpu` via H2D([`src/main.cpp:876`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L876)). This full copy is the spot with a TODO comment saying "not to copy the whole thing unnecessarily"([`src/main.cpp:551`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L551)).

## Attention is finished off by `pagedAttentionKernel`

Prefill built a score matrix, masked the future with `causalMask`([`src/main.cpp:329`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L329)), then called `softmax`([`src/main.cpp:331`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L331)). Decode has neither call. Attention is done in a single `pagedAttention`.

```cpp
            // synchronize block table on cpu with block table on gpu (for attention)
            cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

            pagedAttention(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, buf_2048_1);
```
— [`src/main.cpp:875-878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L875-L878)

`pagedAttentionKernel` takes `kv_cache`, `block_table_gpu`, `gpu_seq_lens`, and `gpu_active_slots` together([`src/kernels.cu:461`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L461)), resolves logical blocks to physical ones through `block_table_gpu`, computes scores while iterating over the blocks([`src/kernels.cu:478-485`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L478-L485)), accumulates the weighted average with warp reduction and online softmax([`src/kernels.cu:486-519`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L486-L519)), and writes the output([`src/kernels.cu:522`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L522)). With no separate mask, **iterating only up to the sequence length** stands in for future masking.

Online softmax is a technique that normalizes attention scores across a row while accumulating the weighted average in a single pass. There's no need to build the whole score matrix per row first, so the result can be produced on the fly while iterating over blocks. A warp is an execution unit where 32 threads move together, and a shuffle is an instruction that exchanges values among those threads — it's used to gather the score sums within a warp. The source comment links online softmax to the [FlashAttention lecture notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)([`src/kernels.cu:473`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L473)). This kernel's warp-shuffle reduction and the internals of its online softmax are part 6's subject.

One thing worth noting: `softmaxDecode`/`softmaxKernelDecode`([`src/kernels.cuh:25`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cuh#L25), [`src/kernels.cu:408-458`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L408-L458)), declared in the decode family, are never called anywhere on the decode path in `src/main.cpp`. Decode's probability normalization is handled by the online softmax inside `pagedAttentionKernel`. This kernel's `MAX_SEQ_LEN`-stride assumption([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413)) presumes a different layout from the prefill softmax's `num_tokens` stride([`src/kernels.cu:262`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L262)), but in the current call graph it's unused code.

## The remaining layers and logits

After attention, the kernels are the same as prefill's. The attention output is written into the same buffer `buf_2048_1` that the input q projection used([`src/main.cpp:878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L878)), and that buffer becomes the O projection's input directly([`src/main.cpp:892`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L892)). Part 4's transpose trick continues unchanged, then `residualAdd`([`src/main.cpp:902`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L902)), post-attn RMSNorm([`src/main.cpp:904`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L904)), SwiGLU's gate·up·`silu`·down([`src/main.cpp:907-969`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L907-L969)), and `residualAdd` again([`src/main.cpp:971`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L971)) end the layer. The only difference from prefill is that the token-count argument is `num_active_slots` instead of `prompt_len`.

After 16 layers, the final RMSNorm([`src/main.cpp:974`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L974)) and the `embed_tokens` logits([`src/main.cpp:976-994`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L976-L994)) come out, and after a D2H copy([`src/main.cpp:996`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L996)) the CPU computes the argmax per active slot and outputs it. Unlike prefill, which picked just the last row([`src/main.cpp:533-543`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L533-L543)), decode picks one per slot.

```cpp
        float max_token = 0.0;
        int max_token_idx = 0;
        for (int slot = 0; slot < num_active_slots; ++slot)
        {
            int active_slot = active_slots[slot];
            max_token = (float)embed_proj_cpu[slot * VOCAB_SIZE]; // TODO: verify if float is good enough in place of nvbf16
            max_token_idx = 0;
            for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
            {
                if ((float)embed_proj_cpu[slot * VOCAB_SIZE + token_idx] > max_token)
                {
                    max_token = embed_proj_cpu[slot * VOCAB_SIZE + token_idx];
                    max_token_idx = token_idx;
                }
            }
```
— [`src/main.cpp:998-1012`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L998-L1012)

The argmax is a CPU loop that scans `VOCAB_SIZE`(128,256) entries per slot(TODO comments [`src/main.cpp:531-532`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L531-L532), `686`). If the generated token is `<|end_of_text|>`(128001) or `<|eot_id|>`(128009), or `current_prompt_len == MAX_SEQ_LEN-1`, the slot is freed and its blocks returned; otherwise the token is accumulated and the length incremented([`src/main.cpp:1015-1037`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L1015-L1037)). That termination branch and slot release are part 7's subject.

## One verification script

`python/` has one small script aimed at the decode family. `decode_test.py` is a tool for turning a prompt into token IDs with a tokenizer.

```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

prompts = {
    "What is 2+2?": [791, 4320, 374, 220, 19],
    "Name a color.": [10544],
    "Say hello.": [9906, 0, 2650, 649, 358, 1520, 499, 449, 4205, 499, 1205, 30],
    "Capital of France?": [791, 6864, 315, 9822, 374, 12366, 13],
}

for prompt, tokens in prompts.items():
    print(f"{prompt} → {t.decode(tokens)}")
```
— [`python/decode_test.py:1-12`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/python/decode_test.py#L1-L12)

The interesting point is that the token lists here actually differ from the chat-template tokens hardcoded in `main.cpp`([`src/main.cpp:585-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L585-L594)). This file isn't code the runtime calls; it's a convenience tool for a person manually checking tokens. This series did not run the file and doesn't claim it works.

## Further reading

Material this repo points to as background for this topic.

- [A visual explanation of RoPE (Fleetwood)](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding) — the meaning of the rotation decode recomputes the angle for every time
- [FlashAttention / online softmax lecture notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — the weighted-average accumulation in `pagedAttentionKernel`
- [PagedAttention paper (Kwon et al., SOSP 2023)](https://arxiv.org/pdf/2309.06180) — the original paper on block-based KV caches
- [CUDA parallel reduction (NVIDIA)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) — the background of warp-shuffle tree reduction

## Limitations of this article

Nothing was built or run, and every description above is a static citation obtained by reading the source at commit `e25bf19`. Three points in particular remain without runtime verification. First, whether the prefill table's Llama 3-scaled angles and decode's unscaled recomputation produce the same rotation is unverified (the recomputation is a TODO, [`src/kernels.cu:375`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L375)). Second, the consistency of the stride-layout assumptions — that `softmaxKernelDecode`'s `MAX_SEQ_LEN` stride([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413)) and the prefill softmax's `num_tokens` stride([`src/kernels.cu:262`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L262)) presume different constants — is also unverified (and that kernel isn't even called). Third, the cost of copying the whole `block_table` H2D per layer([`src/main.cpp:876`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L876), TODO [`src/main.cpp:551`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L551)) wasn't measured. The indexing and buffer sizes depend on compile-time constants such as `BATCH_SIZE=2`([`src/main.cpp:29`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L29)), `MAX_SEQ_LEN=2048`([`src/main.cpp:28`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L28)), and `BLOCK_SIZE=16`([`src/main.cpp:32`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L32)), so if those constants change, every number written here changes with them.
