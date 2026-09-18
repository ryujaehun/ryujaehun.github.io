---
title: "5. 토큰 하나를 만드는 decode 경로 — 왜 커널을 따로 만들었나"
date: "2026-09-17"
draft: false
slug: "code-series-jmaczan--tiny-vllm-05"
categories:
  - code-series
series_id: "jmaczan--tiny-vllm"
series:
  - "jmaczan--tiny-vllm"
series_order: 5
chapter_order: 5
chapter_count: 7
repository: "https://github.com/jmaczan/tiny-vllm"
pinned_commit: "e25bf1994efa90bc98b721ba7c527402f86fbeaf"
---
prefill 이 프롬프트 전체를 한 번에 흘려 보내면, 그다음부터는 지금까지 만든 토큰을 바탕으로 **토큰을 하나씩** 만들어야 한다. 이 단계가 decode 다. 1편에서 본 대로 prefill 과 decode 는 연산의 모양이 달라서 — 전자는 행렬 × 행렬, 후자는 벡터 × 행렬 — 이 저장소는 커널을 따로 만든다. `embeddingGatherKernelDecode`, `ropeKernelDecode`, `softmaxKernelDecode` 처럼 이름에 `Decode` 가 붙은 변형들이고, 여기에 prefill 에 대응이 없는 `pagedAttentionKernel` 이 하나 더 붙는다. 이 편은 그 커널들을 prefill 과 나란히 놓고 **무엇이 다르고 왜 다른 모양인지**, 그리고 새 토큰의 K/V 가 어떻게 캐시에 붙는지를 읽는다.

> 이 시리즈의 모든 인용은 커밋 [`e25bf19`](https://github.com/jmaczan/tiny-vllm/tree/e25bf1994efa90bc98b721ba7c527402f86fbeaf) 기준이다. 필자에게 NVIDIA GPU 가 없어 빌드도 실행도 하지 않았고, 아래 모든 설명은 소스를 읽어 얻은 것이다.

## decode 루프: 어디서 시작하나

decode 는 `prefill` 함수 안이 아니라 `main` 의 `while (true)` 루프([`src/main.cpp:720-1039`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L720-L1039)) 안에서 돈다. 매 반복은 슬롯 상태를 다시 구성하는 것으로 시작한다. `active_slots` 와 `active_tokens` 를 비우고([`src/main.cpp:722-723`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L722-L723)), 빈 슬롯이 있으면 큐의 다음 프롬프트로 `prefill` 을 다시 부르고([`src/main.cpp:724-737`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L724-L737)), 점유 슬롯만 모아 GPU 로 올린다. 이 슬롯·큐 수명은 7편이 다룬다. 여기서 이 편이 따라갈 핵심은 루프 머리에서 만들어 올리는 길이 정보다.

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

길이를 `current_prompt_len + 1` 로 올리는 이유는 나중에 나온다. 레이어 루프 안에서 이번 토큰의 K/V 가 어텐션 호출보다 먼저 캐시에 기록되고([`src/main.cpp:851-873`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L851-L873)), 어텐션 커널은 `gpu_seq_lens` 로 순회할 블록 수를 계산하므로([`src/kernels.cu:470-481`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L470-L481)) 길이에 **이번 토큰까지 포함된 수**가 들어가야 맞다.

## decode 계열 커널과 prefill 과의 대비

decode 루프가 prefill 과 달리 부르는 커널은 네 개다. 셋은 prefill 짝의 변형이고, 하나는 prefill 에 없는 전용 어텐션이다. 구현은 `src/kernels.cu` 의 decode 절([`src/kernels.cu:346-523`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L346-L523))에 prefill 절([`src/kernels.cu:26-344`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L26-L344))과 분리되어 있고, 선언은 커널 헤더 `src/kernels.cuh` 의 decode 절에 모여 있다(3편에서 함께 읽은 그 헤더다).

```cpp
// decode
void embeddingGatherDecode(int *gpu_last_tokens, int num_tokens, __nv_bfloat16 *output, __nv_bfloat16 *embed_tokens);
void ropeDecode(__nv_bfloat16 *input, int position_in_sequence, int proj_dim);
void softmaxDecode(__nv_bfloat16 *input, int seq_len);

// pagedattn
void pagedAttention(int layer, int num_active_slots, __nv_bfloat16 *q_proj, __nv_bfloat16 *kv_cache, int *block_table_gpu, int *gpu_seq_lens, int *gpu_active_slots, __nv_bfloat16 *output);
```
— [`src/kernels.cuh:22-28`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cuh#L22-L28)

두 계열이 어디서 갈리는지 한 표로 보자.

<!-- visual: prefill-decode-contrast supports: [prefill-decode-contrast] -->
| 어느 부분 | prefill | decode | 근거 |
| --- | --- | --- | --- |
| 처리 단위 | 프롬프트의 `prompt_len` 개 토큰을 한 번에 | 활성 슬롯마다 토큰 1개 | [`src/main.cpp:150-158`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L150-L158), [`src/main.cpp:759`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L759) |
| 임베딩 커널 | `embeddingGatherKernel`, `blockIdx.x` = 토큰 위치 | `embeddingGatherKernelDecode`, `blockIdx.x` = 슬롯 | [`src/kernels.cu:32-40`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L32-L40), [`src/kernels.cu:347-356`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L347-L356) |
| RoPE | `ropeKernel_llama3`, 미리 만든 테이블로 모든 위치 회전 | `ropeKernelDecode`, 호출마다 현재 위치 하나만 재계산 | [`src/kernels.cu:173-200`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L173-L200), [`src/kernels.cu:371-384`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L371-L384) |
| 어텐션 전처리 | `causalMask` + `softmax` 호출 | `pagedAttentionKernel` 안의 online softmax | [`src/main.cpp:329-331`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L329-L331), [`src/kernels.cu:509-519`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L509-L519) |
| K/V 기록 | `BLOCK_SIZE` 토큰 묶음을 한 번에 복사 | 토큰 1개 복사, 블록 경계에서만 새 블록 할당 | [`src/main.cpp:251-288`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L251-L288), [`src/main.cpp:851-873`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L851-L873) |
| cuBLAS 의 `n` | `prompt_len` | `num_active_slots` | [`src/main.cpp:182`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L182), [`src/main.cpp:769`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L769) |
| RMSNorm·SwiGLU·residual | 같은 커널, 토큰 수 인자만 다름 | 같은 커널, 토큰 수 인자만 다름 | [`src/kernels.cu:55-81`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L55-L81), [`src/kernels.cu:311-329`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L311-L329), [`src/kernels.cu:331-338`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L331-L338) |

이 표의 두 축이 이 편의 답이다. **decode 계열 커널은 시퀀스 길이 1 을 전제로 인덱싱을 단순화한다.** `embeddingGatherKernelDecode` 는 `blockIdx.x` 를 토큰이 아니라 슬롯으로 쓰고([`src/kernels.cu:349`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L349)), `softmaxKernelDecode` 는 `MAX_SEQ_LEN` 스트라이드 레이아웃을 가정한다([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413)). 그리고 `ropeKernelDecode` 는 전체 위치가 아니라 **현재 위치 하나만** 회전한다. 아래에서 커널별로 본다.

## 임베딩: blockIdx.x 는 슬롯이다

임베딩은 토큰 ID 를 2048 차원 벡터로 바꾸는 첫 단계다. prefill 의 `embeddingGatherKernel` 은 블록 하나가 토큰 하나를 처리한다. `<<<num_input_tokens, 1024>>>` 로 띄우니 `blockIdx.x` 가 곧 토큰 위치고, `workIndex = blockIdx.x * 2048 + threadIdx.x` 로 출력 행을 정한다([`src/kernels.cu:32-45`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L32-L45)). decode 의 `embeddingGatherKernelDecode` 는 모양은 같지만 의미가 다르다. 입력이 토큰 ID 배열이 아니라 슬롯별 마지막 토큰 배열 `gpu_last_tokens` 이고, `blockIdx.x` 가 슬롯을 가리킨다.

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

`input_token = gpu_last_tokens[blockIdx.x]` — `blockIdx.x` 가 슬롯 인덱스다. 출력도 슬롯 순서로 쌓이므로 `workIndex` 의 `blockIdx.x` 도 슬롯이다. 호출부는 `embeddingGatherDecode(gpu_last_tokens, num_active_slots, hidden_state, ...)`([`src/main.cpp:759`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L759))로, `num_active_slots` 가 블록 수가 된다([`src/kernels.cu:361`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L361)). "블록당 토큰 하나" 구조를 그대로 두고 토큰 축을 슬롯 축으로 바꾼 셈이다.

## RoPE: 현재 위치 하나, 매 호출 재계산

RoPE(rotary position embedding)는 위치 정보를 벡터의 회전으로 넣는 기법이다. 위치가 다르면 벡터가 다른 각도만큼 돌아가므로, 회전 각은 위치의 함수다. [시각적 해설](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding)이 저장소 README 가 배경으로 건 자료다.

decode 의 RoPE 는 prefill 과 두 지점에서 갈린다. 첫째, 한 번에 한 위치만 회전한다. `ropeKernelDecode` 는 단일 블록으로 띄워지고([`src/kernels.cu:397`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L397)), 각 스레드가 `2*threadIdx.x` 와 `2*threadIdx.x+1` 의 두 원소를 짝으로 회전한다. 회전 각은 인자로 받은 `position_in_sequence` 하나에 대해서만 계산된다([`src/kernels.cu:378`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L378)). 호출부는 슬롯마다

```cpp
                ropeDecode(&q_proj[slot * EMBEDDING_LENGTH], current_prompt_len[active_slot], EMBEDDING_LENGTH);
                ropeDecode(k_proj_batched_buffer + slot * KV_DIM, current_prompt_len[active_slot], KV_DIM);
```
— [`src/main.cpp:847-848`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L847-L848)

위치 인자는 이 슬롯이 이번에 만들 토큰의 인덱스, 곧 `current_prompt_len` 이다. Q 는 2048 차원, K 는 512 차원이라 `proj_dim` 이 다르다.

둘째, decode 는 각을 매번 다시 계산한다. 이 부분이 이 편에서 가장 주의할 지점이다.

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

`500000.0` 과 `32` 는 하드코딩된 상수다([`src/kernels.cu:376-377`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L376-L377)). 바로 위 주석이 "theta·각·sin/cos 을 미리 계산해서 모든 커널 호출에 재사용하기"를 TODO 로 남겨 둔다([`src/kernels.cu:375`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L375)). 이 재계산은 저자가 만들다 만 지점이다.

대조적으로 prefill 의 `ropeKernel_llama3` 는 `d_cos_table`/`d_sin_table` 을 인자로 받아 `cos_table[token_idx * head_dim + pair_idx * 2]` 를 읽기만 한다([`src/kernels.cu:173-200`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L173-L200)). 이 테이블은 `main` 초입에서 `init_rope_frequencies(HEAD_DIM, MAX_SEQ_LEN, 500000.0f, 32.0f, 1.0f, 4.0f, 8192)` 가 만들어 GPU 로 올린다([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572), 생성과 H2D 복사는 [`src/kernels.cu:96-152`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L96-L152)).

여기서 놓치기 쉬운 차이가 하나 있다. 테이블 경로는 Llama 3 의 주파수 스케일링을 적용한다. `init_rope_frequencies` 는 기본 `inv_freq[i] = 1/pow(rope_theta, 2i/head_dim)` 에서 출발해, 파장이 길어 `original_max_len / low_freq_factor`(=8192)보다 길면 주파수를 `factor`(=32)로 나누고, `high_freq_factor`(=4)와의 사이 구간은 보간한다.

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

decode 의 재계산은 이 스케일링 없이 `theta = 1/pow(500000, 2i/64)` 를 그대로 쓴다. 그래서 저주파 차원(인덱스가 큰 차원)에서는 두 경로의 회전 각이 수치상 같지 않다. prefill 테이블은 Llama 3 스케일링된 각도를, decode 는 스케일 없는 `theta` 를 호출마다 재계산하는 셈이고, 두 경로가 같은 회전을 만드는지는 실행 검증되지 않았다.

## K/V 를 배치 버퍼에 모아 블록에 흩뿌리기

decode 의 K/V 투영은 슬롯을 행으로 묶어 cuBLAS 한 번으로 처리한다. K 투영은 `n = num_active_slots` 으로 호출되어 `k_proj_batched_buffer` 에 쓰고([`src/main.cpp:802-821`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L802-L821)), V 도 같은 모양으로 `v_proj_batched_buffer` 에 쓴다([`src/main.cpp:824-842`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L824-L842)). 두 버퍼는 decode 전용으로 `BATCH_SIZE × KV_DIM` 크기로 따로 잡혀 있다([`src/main.cpp:683-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L683-L693)). 전치 트릭 자체는 4편에서 본 골격 그대로다 — 바뀌는 것은 `n` 이 `prompt_len` 에서 슬롯 수로 줄어든다는 것뿐. 슬롯별 RoPE([`src/main.cpp:844-849`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L844-L849))를 지나면, 새 토큰의 K/V 를 paged KV cache 의 해당 블록에 복사한다. paged KV cache 는 KV 를 16 토큰 단위의 작은 블록으로 쪼개 보관하고 block table 로 "어느 시퀀스의 몇 번째 묶음이 어느 블록인지"를 추적하는 구조다(1편에서 본 2GB `cudaMalloc` 을 그런 블록들로 쪼갠 것이다). 블록 레이아웃과 인덱싱은 6편이 다룬다.

복사가 prefill 과 다른 지점은 **한 번에 한 토큰만** 다룬다는 것. 슬롯마다 새 토큰의 위치를 `current_prompt_len` 으로 삼아 논리 블록과 블록 내 오프셋으로 나눈다.

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

새 블록 할당은 오프셋이 0, 곧 새 블록의 첫 토큰일 때만 일어난다([`src/main.cpp:859-865`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L859-L865)). 아니면 이미 배정된 블록을 그대로 쓴다. K 는 블록 시작 + `token_in_block_idx * KV_DIM * sizeof(__nv_bfloat16)` 위치에, V 는 그 자리에 `V_OFFSET` 을 더한 위치에 D2D `cudaMemcpy` 로 기록한다. `V_OFFSET = BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16)` 이고([`src/main.cpp:33`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L33)), 블록 하나는 K 와 V 를 합쳐 `BLOCK_BYTES` 를 차지한다([`src/main.cpp:34`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L34)).

prefill 은 같은 일을 `BLOCK_SIZE` 토큰 묶음 단위로 한다. 토큰 구간마다 블록을 할당하고 묶음 전체를 한 번에 복사한다([`src/main.cpp:251-288`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L251-L288)). decode 는 그 묶음이 토큰 하나로 줄어든 형태다. 기록 후에는 `block_table` 전체를 `block_table_gpu` 로 H2D 동기화한다([`src/main.cpp:876`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L876)). 이 전체 복사는 "불필요하게 전체를 복사하지 않게" 하는 TODO 주석이 달린 지점이다([`src/main.cpp:551`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L551)).

## 어텐션은 pagedAttentionKernel 이 끝낸다

prefill 은 점수 행렬을 만들고 `causalMask`([`src/main.cpp:329`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L329))로 미래를 가린 뒤 `softmax`([`src/main.cpp:331`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L331))를 호출했다. decode 는 그 두 호출이 없다. 어텐션은 `pagedAttention` 한 번으로 끝난다.

```cpp
            // synchronize block table on cpu with block table on gpu (for attention)
            cudaMemcpy(block_table_gpu, block_table.data(), MAX_SEQUENCES * N_LAYERS * MAX_BLOCKS_PER_SEQ * sizeof(int), cudaMemcpyHostToDevice);

            pagedAttention(layer, num_active_slots, q_proj, kv_cache, block_table_gpu, gpu_seq_lens, gpu_active_slots, buf_2048_1);
```
— [`src/main.cpp:875-878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L875-L878)

`pagedAttentionKernel` 은 `kv_cache`, `block_table_gpu`, `gpu_seq_lens`, `gpu_active_slots` 를 함께 받아([`src/kernels.cu:461`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L461)) `block_table_gpu` 를 거쳐 논리 블록을 물리 블록으로 바꾸고 블록을 순회하며 점수를 계산하고([`src/kernels.cu:478-485`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L478-L485)), 워프 리덕션과 online softmax 로 가중 평균을 누적해([`src/kernels.cu:486-519`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L486-L519)) 출력을 쓴다([`src/kernels.cu:522`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L522)). 별도 마스크 없이 **시퀀스 길이까지만 순회**하는 것이 미래 마스킹을 대신한다.

online softmax 는 어텐션 점수를 행 전체에 대해 정규화하면서 가중 평균을 한 번에 누적하는 기법이다. 행마다 점수 행렬을 먼저 다 만들 필요가 없어서 블록을 순회하며 곧바로 결과를 만들 수 있다. 워프(warp)는 32 개 스레드가 함께 움직이는 실행 단위이고, 셔플(shuffle)은 그 스레드들끼리 값을 주고받는 명령이다 — 점수 합을 워프 안에서 모을 때 쓰인다. 소스 주석이 online softmax 를 [FlashAttention 강의노트](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)로 연결해 둔다([`src/kernels.cu:473`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L473)). 이 커널의 워프 셔플 리덕션과 online softmax 의 내부는 6편이 다룬다.

한 가지 짚을 게 있다. decode 계열로 선언된 `softmaxDecode`/`softmaxKernelDecode`([`src/kernels.cuh:25`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cuh#L25), [`src/kernels.cu:408-458`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L408-L458))는 `src/main.cpp` 의 decode 경로 어디에서도 호출되지 않는다. decode 의 확률 정규화는 `pagedAttentionKernel` 안의 online softmax 가 맡기 때문이다. 이 커널의 `MAX_SEQ_LEN` 스트라이드 가정([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413))은 prefill softmax 의 `num_tokens` 스트라이드([`src/kernels.cu:262`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L262))와 다른 레이아웃을 전제하는데, 현재 호출 그래프에서는 쓰이지 않는 코드다.

## 남은 레이어와 로짓

어텐션 뒤는 prefill 과 같은 커널이다. 어텐션 출력은 입력 q 투영과 같은 버퍼 `buf_2048_1` 에 쓰이고([`src/main.cpp:878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L878)), 그 버퍼가 곧바로 O 투영의 입력이 된다([`src/main.cpp:892`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L892)). 4편의 전치 트릭이 그대로 이어지고, 이후 `residualAdd`([`src/main.cpp:902`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L902)), post-attn RMSNorm([`src/main.cpp:904`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L904)), SwiGLU 의 gate·up·`silu`·down([`src/main.cpp:907-969`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L907-L969)), 다시 `residualAdd`([`src/main.cpp:971`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L971))로 한 레이어가 끝난다. prefill 과의 차이는 토큰 수 인자로 `prompt_len` 대신 `num_active_slots` 를 받는다는 것뿐.

16개 레이어를 지나면 최종 RMSNorm([`src/main.cpp:974`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L974))과 `embed_tokens` 로짓([`src/main.cpp:976-994`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L976-L994))이 나오고, D2H 복사([`src/main.cpp:996`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L996)) 뒤 CPU 가 활성 슬롯별 argmax 를 구해 출력한다. prefill 이 마지막 행 하나만 고른 것([`src/main.cpp:533-543`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L533-L543))과 달리 decode 는 슬롯마다 하나씩 고른다.

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

argmax 는 슬롯마다 `VOCAB_SIZE`(128,256)만큼 훑는 CPU 루프다(TODO 주석 [`src/main.cpp:531-532`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L531-L532), `686`). 생성 토큰이 `<|end_of_text|>`(128001) 또는 `<|eot_id|>`(128009) 이거나 `current_prompt_len == MAX_SEQ_LEN-1` 이면 슬롯을 비우고 블록을 반납하며, 아니면 토큰을 누적하고 길이를 1 올린다([`src/main.cpp:1015-1037`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L1015-L1037)). 이 종료 분기와 슬롯 해제는 7편이 다룬다.

## 검증용 스크립트 하나

`python/` 에는 decode 계열을 겨냥한 작은 스크립트가 하나 있다. `decode_test.py` 는 토크나이저로 프롬프트를 토큰 ID 로 바꿔 보는 도구다.

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

흥미로운 점은 여기 토큰 목록과 `main.cpp` 에 하드코딩된 채팅 템플릿 토큰([`src/main.cpp:585-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L585-L594))이 실제로 다르다는 것이다. 이 파일은 런타임이 부르는 코드가 아니라 사람이 손으로 토큰을 확인할 때 쓰는 편의 도구다. 이 시리즈는 이 파일을 실행하지 않았고, 동작을 주장하지도 않는다.

## 더 읽을거리

이 저장소가 이 주제의 배경으로 건 자료다.

- [RoPE 시각적 해설 (Fleetwood)](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding) — decode 가 매번 각을 다시 계산하는 그 회전의 의미
- [FlashAttention / online softmax 강의노트](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — `pagedAttentionKernel` 의 가중 평균 누적
- [PagedAttention 논문 (Kwon et al., SOSP 2023)](https://arxiv.org/pdf/2309.06180) — 블록 단위 KV cache 의 원 논문
- [CUDA 병렬 리덕션 (NVIDIA)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) — 워프 셔플 트리 리덕션의 배경

## 이 글의 한계

빌드도 실행도 하지 않았고, 위 모든 설명은 커밋 `e25bf19` 의 소스를 읽어 얻은 정적 인용이다. 특히 세 지점은 실행 검증 없이 남는다. 첫째, prefill 테이블의 Llama 3 스케일링된 각도와 decode 의 스케일 없는 재계산이 같은 회전을 만드는지는 검증되지 않았고(재계산은 TODO, [`src/kernels.cu:375`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L375)), 둘째, `softmaxKernelDecode` 의 `MAX_SEQ_LEN` 스트라이드([`src/kernels.cu:413`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L413))와 prefill softmax 의 `num_tokens` 스트라이드([`src/kernels.cu:262`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/kernels.cu#L262))가 서로 다른 상수를 전제한다는 스트라이드 레이아웃 가정의 정합성도 검증되지 않았다(이 커널은 호출되지도 않는다). 셋째, 레이어마다 `block_table` 전체를 H2D 로 복사하는 비용([`src/main.cpp:876`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L876), TODO [`src/main.cpp:551`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L551))은 측정하지 않았다. 인덱싱과 버퍼 크기는 `BATCH_SIZE=2`([`src/main.cpp:29`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L29)), `MAX_SEQ_LEN=2048`([`src/main.cpp:28`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L28)), `BLOCK_SIZE=16`([`src/main.cpp:32`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L32)) 같은 컴파일 타임 상수에 의존하므로, 상수가 바뀌면 여기 적은 모든 수치가 함께 바뀐다.
