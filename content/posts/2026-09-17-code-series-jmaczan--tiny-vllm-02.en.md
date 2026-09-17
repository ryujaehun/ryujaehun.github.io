---
title: "2. Weight Loading and GPU Buffer Design"
date: "2026-09-17"
draft: false
slug: "code-series-jmaczan--tiny-vllm-02"
categories:
  - code-series
series_id: "jmaczan--tiny-vllm"
chapter_order: 2
chapter_count: 7
repository: "https://github.com/jmaczan/tiny-vllm"
pinned_commit: "e25bf1994efa90bc98b721ba7c527402f86fbeaf"
---
In this engine, right after creating the cuBLAS handle, `main()` immediately does two things ([`src/main.cpp:557-569`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L557-L569)): it takes the weights out of `model.safetensors` and puts them on the GPU, and it preallocates the buffers that every computation from here on will use. This chapter reads both.

The point of both is that they happen **exactly once**. `cudaMalloc` and host-to-device copies are too expensive to repeat on every request, so the weights go up in one piece and the buffers are all grabbed at the start, then reused under different names.

First, settle on the common unit for sizing. This engine stores both activations and weights as bfloat16. bfloat16 is a 16-bit floating-point format with 1 sign bit, 8 exponent bits, and 7 mantissa bits. At half the size of float (4 bytes) it uses half the memory while keeping float's exponent range, which is why it is the standard for LLM training and inference. Every byte calculation in this chapter starts from "one element = 2 bytes." The details of the format are left to the [README's bfloat16 explanation](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/README.md).

This chapter covers two of those steps, `loadWeights` ([`src/main.cpp:79-147`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L79-L147)) and the buffer-allocation block ([`src/main.cpp:617-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L693)). The same initialization block also builds the RoPE frequency table ([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572)) and the request queue ([`src/main.cpp:583-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L583-L594)), but those belong to chapters 3 and 7 respectively.

## One safetensors file, loaded onto the GPU in one piece

`loadWeights` really does one thing: it opens `model.safetensors` and uploads every weight byte in it to the GPU. safetensors is Hugging Face's weight storage format. Because it is a pure blob of bytes that cannot carry executable code, it is safer than pickle-based `.bin` files, and the file is laid out as "8-byte header length → JSON header → weight bytes." The JSON header holds each tensor's name and its byte range within the file. ([format docs](https://huggingface.co/docs/safetensors/en/index))

The function splits into four steps. First, `checkGPUStatus` ([`src/main.cpp:40-62`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L40-L62)) checks whether a device exists and, if not, prints an error and `return 1`s ([`src/main.cpp:44-48`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L44-L48)). Then it opens the file. The path is written as a literal string rather than an argument — the first sign of the assumption that the model architecture is fixed.

```cpp
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary); // TODO: use args to provide the path or smth
if (!safetensors_file.is_open())
{
    std::cout << "Can't open model.safetensors file\n";
    safetensors_file.close();
    return 1;
}
```
— [`src/main.cpp:87-93`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L87-L93)

Third, the header is read. The first 8 bytes of the file are the header's length, and the header string is read by that length.

```cpp
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
// READ SAFETENSORS HEADER
std::string header;
header.resize(header_size);
safetensors_file.read(header.data(), header_size);
```
— [`src/main.cpp:96-101`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L96-L101)

The safetensors header is JSON. The `include/json.hpp` (nlohmann/json) we saw in chapter 1 is used exactly here.

The fourth step is the core. Walking the parsed JSON, it reads the two `data_offsets` values per tensor: the end value (index 1) tracks the overall end of the weights in `max_offset`, and the start value (index 0) is stored in a name→offset map. `__metadata__` is a metadata key with no actual data, so it is skipped.

```cpp
std::unordered_map<std::string, uint64_t> offsets;
json header_json = json::parse(header);
uint64_t max_offset = 0;
for (auto &[key, value] : header_json.items())
{
    if (key == "__metadata__")
    {
        continue;
    }
    uint64_t offset_end = value["data_offsets"].at(1).get<uint64_t>();
    if (offset_end > max_offset)
    {
        max_offset = offset_end;
    }
    offsets[key] = value["data_offsets"].at(0).get<uint64_t>();
}
```
— [`src/main.cpp:103-118`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L103-L118)

This `max_offset` is exactly the byte count of all the weights. This is where the function's design shows itself: instead of a `cudaMalloc` per tensor, it grabs **all weight bytes at once**. It makes a CPU vector of the same size, reads the rest of the file into it wholesale, and uploads it with a single host-to-device copy.

```cpp
void *model_weights;
cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

std::vector<char> model_weights_cpu;
model_weights_cpu.resize(max_offset);
safetensors_file.read(model_weights_cpu.data(), max_offset);

cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
```
— [`src/main.cpp:120-127`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L120-L127)

So the GPU ends up with a single chunk rather than per-tensor allocations, and each tensor is accessed as a byte range inside that chunk.

## The `Weights` struct — tensor names hardcoded as strings

The key for finding a tensor inside the chunk is its name. `loadWeights` looks up the name→offset map and fills the pointer fields of the `Weights` struct.

```cpp
struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};
```
— [`src/main.cpp:64-77`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L64-L77)

That is all: two layer-independent fields (`embed_tokens`, `norm`) and nine per-layer arrays. With nine kinds across 16 layers, the struct references 2 + 9×16 = 146 tensors. That number is not the total number of tensors in the file, but the number this struct points to.

The mapping loop assembles each tensor name as a string.

```cpp
weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
for (int i = 0; i < N_LAYERS; ++i)
{
    weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
    ...
}
```
— [`src/main.cpp:132-145`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L132-L145) (excerpt)

The name strings fix the model architecture. Any other model has different tensor names, so this mapping breaks — which is why this engine always runs assuming Llama 3.2 1B-Instruct.

## Preallocating the 16 compute buffers

The principle behind the buffer design appeared in chapter 1: `cudaMalloc` is expensive, so grab everything at once and reuse it. The allocations all live in one block, [`src/main.cpp:617-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L693). Their sizes are fixed by the compile-time constants at the top of the file ([`src/main.cpp:16-31`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L16-L31)). The key one is `MAX_BUFFER_SIZE`.

```cpp
constexpr int EMBEDDING_LENGTH = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int KV_DIM = 512;
constexpr int NUM_Q_HEADS = 32;
constexpr int VOCAB_SIZE = 128256;
constexpr int BATCH_SIZE = 2;
constexpr int MAX_PROMPT_LEN = 512;
constexpr int MAX_BUFFER_SIZE = std::max(MAX_PROMPT_LEN, BATCH_SIZE);
```
— [`src/main.cpp:16-31`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L16-L31) (excerpt)

`MAX_BUFFER_SIZE` is `max(512, 2)`, so 512. Prefill handles a 512-token prompt and decode handles 2 batch slots; the shared buffers that serve both shapes are sized to the larger one, 512.

Every allocation follows one pattern — one `cudaMalloc` per buffer. A sample:

```cpp
__nv_bfloat16 *hidden_state;
cudaMalloc(&hidden_state, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
...
__nv_bfloat16 *buf_2048_1; // shared between q_proj and attn_scores_v
cudaMalloc(&buf_2048_1, MAX_BUFFER_SIZE * sizeof(__nv_bfloat16) * EMBEDDING_LENGTH);
...
__nv_bfloat16 *prefill_attn_scores;
cudaMalloc(&prefill_attn_scores, MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * NUM_Q_HEADS);
...
__nv_bfloat16 *embed_proj;
cudaMalloc(&embed_proj, sizeof(__nv_bfloat16) * MAX_BUFFER_SIZE * VOCAB_SIZE);
```
— [`src/main.cpp:622-677`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L622-L677) (excerpt)

Here are the 16 buffers allocated in this block.

<!-- visual: buffer-allocation-table supports: [buffer-allocation-table] -->
| Buffer | Size | Bytes (bf16 = 2B) | Role | Source |
| --- | --- | --- | --- | --- |
| `gpu_input_tokens` | `MAX_PROMPT_LEN × sizeof(int)` | 2 KiB | Temporary buffer for moving prompt token IDs to the GPU | [`src/main.cpp:617-618`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L618) |
| `input_embeddings` | `MAX_PROMPT_LEN × EMBEDDING_LENGTH × 2` | 2 MiB | Token embedding output | [`src/main.cpp:619-620`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L619-L620) |
| `hidden_state` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | Layer input/output, residual accumulation | [`src/main.cpp:622-623`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L622-L623) |
| `rms_norms` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | RMSNorm output | [`src/main.cpp:625-626`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L625-L626) |
| `buf_2048_1` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | Shared between `q_proj` and `attn_scores_v` | [`src/main.cpp:628-629`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L628-L629) |
| `k_proj_temp_buf` | `MAX_PROMPT_LEN × KV_DIM × 2` | 512 KiB | Prefill K projection temporary buffer | [`src/main.cpp:635-636`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L635-L636) |
| `v_proj_temp_buf` | `MAX_PROMPT_LEN × KV_DIM × 2` | 512 KiB | Prefill V projection temporary buffer | [`src/main.cpp:638-639`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L638-L639) |
| `prefill_attn_scores` | `MAX_PROMPT_LEN² × NUM_Q_HEADS × 2` | 16 MiB | Prefill attention scores | [`src/main.cpp:647-648`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L647-L648) |
| `buf_2048_2` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | Shared between `o_proj` and `down` | [`src/main.cpp:656-657`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L656-L657) |
| `gate` | `MAX_BUFFER_SIZE × HIDDEN_DIM × 2` | 8 MiB | SwiGLU gate output | [`src/main.cpp:662-663`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L662-L663) |
| `up` | `MAX_BUFFER_SIZE × HIDDEN_DIM × 2` | 8 MiB | SwiGLU up output | [`src/main.cpp:667-668`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L667-L668) |
| `embed_proj` | `MAX_BUFFER_SIZE × VOCAB_SIZE × 2` | 125.25 MiB | Logits | [`src/main.cpp:676-677`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L676-L677) |
| `embed_proj_cpu` | `MAX_BUFFER_SIZE × VOCAB_SIZE` elements (host) | 125.25 MiB | Logits D2H receive | [`src/main.cpp:681-682`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L681-L682) |
| `gpu_last_tokens` | `BATCH_SIZE × sizeof(int)` | 8 B | Tokens just before decode | [`src/main.cpp:684-685`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L684-L685) |
| `k_proj_batched_buffer` | `BATCH_SIZE × KV_DIM × 2` | 2 KiB | Decode K batch | [`src/main.cpp:689-690`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L689-L690) |
| `v_proj_batched_buffer` | `BATCH_SIZE × KV_DIM × 2` | 2 KiB | Decode V batch | [`src/main.cpp:692-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L692-L693) |

The byte counts in the table are computed from the constants. Let's do a few by hand. `prefill_attn_scores` is 512 × 512 × 32 = 8,388,608 elements, and as bf16 that is 16,777,216 bytes, or 16 MiB. `gate` and `up` are each 512 × 8192 = 4,194,304 elements, or 8 MiB; `embed_proj` is 512 × 128,256 = 65,667,072 elements, or 125.25 MiB. The vocabulary size `VOCAB_SIZE = 128256` dominates the logit buffer. And `embed_proj` is sized by `MAX_BUFFER_SIZE` (512) rather than the batch size, so the prompt length fixes this buffer. Conversely, the four shared buffers `hidden_state`, `rms_norms`, `buf_2048_1`, `buf_2048_2` and `input_embeddings` are all 2 MiB because `MAX_PROMPT_LEN` and `MAX_BUFFER_SIZE` happen to be equal.

Only the logit buffer has a host-side counterpart, `embed_proj_cpu`. The reason is that the argmax runs on the CPU. After the D2H copy, it scans the last token's row for the index of the largest value.

```cpp
cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * prompt_len * VOCAB_SIZE, cudaMemcpyDeviceToHost);
// argmax to get the output token
// TODO: write a proper kernel for it
int last_token_offset = (prompt_len - 1) * VOCAB_SIZE;
float max_token = (float)embed_proj_cpu[last_token_offset];
int max_token_idx = 0;
for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
{
    if ((float)embed_proj_cpu[token_idx + last_token_offset] > max_token)
    {
        max_token = embed_proj_cpu[token_idx + last_token_offset];
        max_token_idx = token_idx;
    }
}
```
— [`src/main.cpp:529-543`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L529-L543)

A CPU loop over all of `VOCAB_SIZE`, running every token, is expensive. The code knows it too — there is a "write a proper kernel for it" TODO on it. Moving the argmax to the GPU would make this D2H copy and the CPU vector disappear entirely.

The three buffers at the end of the block are decode-only. `gpu_last_tokens` holds the token IDs just before decode, and `k_proj_batched_buffer`/`v_proj_batched_buffer` are temporary buffers that gather the per-slot K/V projections at once. At `BATCH_SIZE × KV_DIM`, they are very small — 2 KiB.

```cpp
// decode-only allocation
int *gpu_last_tokens;
cudaMalloc(&gpu_last_tokens, BATCH_SIZE * sizeof(int));
// TODO: move argmax to GPU and get rid of these CPU<->GPU tokens moves
// reused temporary buffers for K and V cache computation during decode
__nv_bfloat16 *k_proj_batched_buffer;
cudaMalloc(&k_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);

__nv_bfloat16 *v_proj_batched_buffer;
cudaMalloc(&v_proj_batched_buffer, BATCH_SIZE * sizeof(__nv_bfloat16) * KV_DIM);
```
— [`src/main.cpp:683-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L683-L693)

Prefill's K/V temporary buffers (`k_proj_temp_buf`, `v_proj_temp_buf`, 512 KiB) are separate from decode's batched buffers (2 KiB) because the two paths handle different shapes: prefill pushes the whole prompt at once, decode handles one token per batch slot. Chapter 5 looks at that contrast.

## One buffer, two roles — the alias design

Look at the two buffers marked "shared" in the table. `buf_2048_1` backs both the q projection and the attention scores×V result; `buf_2048_2` backs both the O projection and the SwiGLU down projection result. This is the classic GPU memory-saving trick — they are used at different points in time, so one chunk is enough.

Here is the order in which each gets reused inside prefill.

```cpp
q_proj = buf_2048_1;
...
attn_scores_v = buf_2048_1;
```
— [`src/main.cpp:177`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L177), [`src/main.cpp:343`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L343)

First the Q projection writes its result into `buf_2048_1`. Then the per-head attention-score computation ([`src/main.cpp:301-327`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L301-L327)) reads all of that Q. Only after that does line 343 reuse the same buffer as the scores×V output. Since the buffer is not overwritten while Q is still alive, it is safe — but that safety rests only on the static fact of the "read order" visible in the source; it is not verified by running the code.

The same goes for `buf_2048_2`.

```cpp
o_proj = buf_2048_2;
...
down = buf_2048_2;
```
— [`src/main.cpp:377`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L377), [`src/main.cpp:470`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L470)

At line 377 it is reused as `o_proj = buf_2048_2`, and the following cuBLAS call ([`src/main.cpp:378-396`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L378-L396)) writes the O projection output. `residualAdd` ([`src/main.cpp:399`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L399)) reads that value and adds it into `hidden_state`; only then is the same buffer reused at line 470 as the down projection output. The lifetimes of the O and down results never overlap.

The aliasing continues on the decode path. `pagedAttention`'s output is again written into `buf_2048_1` ([`src/main.cpp:878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L878)) and immediately becomes the O projection's input. Why this in-place write is safe is chapter 6's subject.

## What else is set up in the initialization block

Besides the 16 compute buffers, three things are created in the same initialization block.

- The paged KV cache allocator ([`src/main.cpp:574-581`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L574-L581)) — the 2 GB `kv_cache`, `free_blocks` (0..65535), and `block_table` (all `-1`). PagedAttention is the technique of managing the KV cache as small blocks (pages) rather than one large contiguous region. Chapter 1 showed its setup, and the attention kernel that reads and writes this structure is chapter 6's subject. ([Kwon et al., SOSP 2023](https://arxiv.org/pdf/2309.06180))
- The RoPE frequency table ([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572)) — chapter 3's subject.
- The request queue ([`src/main.cpp:583-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L583-L594)) — chapter 7's subject.

What this chapter does look at is the slot and batching state.

```cpp
// BATCH
std::vector<bool> is_slot_free(BATCH_SIZE, true); // set to false when slot taken, set to true when free

std::vector<std::vector<int>> generated_tokens(BATCH_SIZE);
std::vector<int> last_generated_tokens(BATCH_SIZE);
std::vector<int> current_prompt_len(BATCH_SIZE, 0);
...
int *gpu_active_slots;
cudaMalloc(&gpu_active_slots, BATCH_SIZE * sizeof(int));
int *gpu_seq_lens;
cudaMalloc(&gpu_seq_lens, BATCH_SIZE * sizeof(int));
```
— [`src/main.cpp:597-610`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L597-L610) (excerpt)

`is_slot_free` remembers whether a slot is free, and `current_prompt_len` the slot's current length. `gpu_active_slots` and `gpu_seq_lens` are contiguous memory that hands the list of active slots and their lengths to the kernels in the decode loop. When slots empty and refill is chapter 7's subject.

## Further reading

- [README](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/README.md) — the "How floating-point numbers work and why we use bfloat16" section explains this chapter's bf16 unit, and the "GPU and CPU memory" section explains why the buffers are preallocated on the GPU.
- [safetensors docs (Hugging Face)](https://huggingface.co/docs/safetensors/en/index) — the header JSON + byte blob structure
- [PagedAttention paper (Kwon et al., SOSP 2023)](https://arxiv.org/pdf/2309.06180) — the background for cutting the 2 GB cache into pages (background for chapter 6)

## Limits of this chapter

This series was written in an environment without an NVIDIA GPU or CUDA toolchain, so nothing — `cudaMalloc`, `cudaMemcpy`, or anything else — was run. Everything about `loadWeights`' behavior and the byte counts in the table is a source citation, and the byte counts are computed from compile-time constants rather than measured. The real `model.safetensors` holds the weights of the gated model `meta-llama/Llama-3.2-1B-Instruct`, so obtaining it is itself a constraint, and loading it has not been verified. Because the tensor names are hardcoded as strings, the mapping breaks on any other model architecture. The safety of the buffer aliases is a static claim about read order, with no execution verification of concurrency or races. If a constant changes (`MAX_SEQ_LEN`, `BATCH_SIZE`, `MAX_PROMPT_LEN`, etc.), every size in the table changes with it.
