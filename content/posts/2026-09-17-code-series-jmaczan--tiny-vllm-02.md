---
title: "2. 가중치 적재와 GPU 버퍼 설계"
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
이 엔진에서 `main()` 은 cuBLAS 핸들을 만든 뒤 곧바로 두 가지를 한다([`src/main.cpp:557-569`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L557-L569)). `model.safetensors` 파일에서 가중치를 꺼내 GPU 메모리에 올리는 일, 그리고 앞으로 모든 계산이 쓸 버퍼를 미리 잡아 두는 일. 이번 편은 그 둘을 읽는다.

둘 다 "딱 한 번"만 하는 게 핵심이다. `cudaMalloc` 이나 Host-to-Device 복사는 매 요청마다 반복하기엔 비싼 작업이라, 가중치는 한 번에 통째로 올리고 버퍼는 처음에 전부 잡아 둔 뒤 이름만 바꿔 가며 재사용한다.

크기를 계산할 공통 단위부터 정하자. 이 엔진은 활성과 가중치를 모두 bfloat16 으로 저장한다. bfloat16 은 부호 1비트, 지수 8비트, 가수 7비트짜리 16비트 부동소수점이다. float(4바이트)의 절반 크기라 메모리를 반으로 쓰면서도 지수 범위는 float 와 같아, LLM 의 훈련과 추론에서 표준으로 쓰인다. 이 편의 모든 바이트 계산은 "요소 하나 = 2바이트"에서 출발한다. 형식의 세부는 [README 의 bfloat16 설명](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/README.md)으로 넘긴다.

이 편이 다루는 건 그중 두 단계, `loadWeights`([`src/main.cpp:79-147`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L79-L147))와 버퍼 할당 구간([`src/main.cpp:617-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L693))이다. 같은 초기화 구간에는 RoPE 주파수 테이블([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572))과 요청 큐([`src/main.cpp:583-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L583-L594))도 만들지만, 그것은 각각 3편과 7편의 몫이다.

## safetensors 파일 하나를 통째로 GPU 로 올린다

`loadWeights` 가 하는 일은 결국 하나다. `model.safetensors` 를 열어 그 안의 가중치 바이트를 전부 GPU 에 올린다. safetensors 는 Hugging Face 의 가중치 저장 포맷이다. 실행 코드를 담을 수 없는 순수 바이트 덩어리라 pickle 계열 `.bin` 보다 안전하고, 파일은 "헤더 길이 8바이트 → JSON 헤더 → 가중치 바이트" 순서로 생겼다. JSON 헤더가 각 텐서의 이름과 파일 내 바이트 범위를 담는다. ([형식 문서](https://huggingface.co/docs/safetensors/en/index))

함수는 네 단계로 나뉜다. 먼저 `checkGPUStatus`([`src/main.cpp:40-62`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L40-L62))로 디바이스가 있는지 보고, 없으면 오류를 출력하고 `return 1` 한다([`src/main.cpp:44-48`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L44-L48)). 그다음 파일을 연다. 경로가 인자가 아니라 문자열 그대로 적혀 있는 것부터 모델 구조가 고정돼 있다는 전제의 첫 징표다.

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

세 번째는 헤더 읽기다. 파일 앞 8바이트가 헤더의 길이이고, 그 길이만큼 헤더 문자열을 읽는다.

```cpp
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
// READ SAFETENSORS HEADER
std::string header;
header.resize(header_size);
safetensors_file.read(header.data(), header_size);
```
— [`src/main.cpp:96-101`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L96-L101)

safetensors 헤더는 JSON 이다. 1편에서 본 `include/json.hpp`(nlohmann/json)가 정확히 여기서 쓰인다.

네 번째가 핵심이다. 파싱한 JSON 을 돌며 텐서마다 `data_offsets` 의 두 값을 읽는데, 끝값(인덱스 1)으로 가중치의 전체 끝을 `max_offset` 에 누적하고 시작값(인덱스 0)을 이름→오프셋 맵에 저장한다. `__metadata__` 는 실제 데이터가 없는 메타 키라 건너뛴다.

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

이 `max_offset` 이 곧 가중치 전체의 바이트 수다. 여기서부터가 이 함수의 설계다. 텐서마다 `cudaMalloc` 을 하지 않고, **가중치 바이트 전체를 한 번에** 잡는다. 같은 크기의 CPU 벡터를 만들어 파일의 나머지를 통째로 읽고, Host-to-Device 복사 한 번으로 올린다.

```cpp
void *model_weights;
cudaMalloc(&model_weights, max_offset); // max_offset tells where the model weights end in the memory

std::vector<char> model_weights_cpu;
model_weights_cpu.resize(max_offset);
safetensors_file.read(model_weights_cpu.data(), max_offset);

cudaMemcpy(model_weights, model_weights_cpu.data(), max_offset, cudaMemcpyHostToDevice);
```
— [`src/main.cpp:120-127`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L120-L127)

이렇게 하면 GPU 에 텐서별 할당이 아니라 한 덩어리만 남는다. 각 텐서는 그 덩어리 위의 주소 범위로 접근한다.

## Weights 구조체 — 텐서 이름이 문자열로 고정돼 있다

덩어리에서 텐서를 찾는 기준은 이름이다. `loadWeights` 는 이름→오프셋 맵을 뒤져 `Weights` 구조체의 포인터 필드를 채운다.

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

레이어와 무관한 두 필드(`embed_tokens`, `norm`)와 레이어별 아홉 배열이 전부다. 레이어별 아홉 종류가 16 레이어씩 있으니, 이 구조체가 참조하는 텐서는 2 + 9×16 = 146 개다. 이 숫자는 파일 안에 있는 텐서 전체 수가 아니라, 이 구조체가 가리키는 텐서 수다.

매핑 루프는 텐서 이름을 문자열로 조립한다.

```cpp
weights.embed_tokens = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.embed_tokens.weight"));
weights.norm = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.norm.weight"));
for (int i = 0; i < N_LAYERS; ++i)
{
    weights.input_layernorm[i] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"));
    ...
}
```
— [`src/main.cpp:132-145`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L132-L145) (일부 발췌)

이름 문자열이 모델 구조를 고정한다. 다른 모델은 텐서 이름이 다르니 이 매핑이 어긋난다 — 이 엔진이 항상 Llama 3.2 1B-Instruct 를 전제로 돌아가는 이유다.

## 계산 버퍼 16개를 한 번에 잡아 둔다

버퍼 설계의 원칙은 1편에서 봤다. `cudaMalloc` 은 비싸니 한 번에 전부 잡고 재사용한다. 할당은 [`src/main.cpp:617-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L693) 한 구간에 모여 있다. 크기는 파일 맨 위의 컴파일 타임 상수([`src/main.cpp:16-31`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L16-L31))가 정한다. 그중 핵심은 `MAX_BUFFER_SIZE` 다.

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
— [`src/main.cpp:16-31`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L16-L31) (일부 발췌)

`MAX_BUFFER_SIZE` 는 `max(512, 2)` 라 512 다. prefill 은 프롬프트 512 토큰을, decode 는 배치 2 슬롯을 다루는데, 두 모양을 아우르는 공용 버퍼는 더 큰 512 기준으로 잡는다.

할당은 전부 한 패턴이다 — 버퍼 하나에 `cudaMalloc` 하나. 대표만 보면:

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
— [`src/main.cpp:622-677`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L622-L677) (일부 발췌)

이 구간에서 잡는 버퍼 16개를 정리하면 다음과 같다.

<!-- visual: buffer-allocation-table supports: [buffer-allocation-table] -->
| 버퍼 | 크기 식 | 바이트 (bf16=2B) | 역할 | 출처 |
| --- | --- | --- | --- | --- |
| `gpu_input_tokens` | `MAX_PROMPT_LEN × sizeof(int)` | 2 KiB | 프롬프트 토큰 ID 를 GPU 로 올리는 임시 버퍼 | [`src/main.cpp:617-618`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L617-L618) |
| `input_embeddings` | `MAX_PROMPT_LEN × EMBEDDING_LENGTH × 2` | 2 MiB | 토큰 임베딩 결과 | [`src/main.cpp:619-620`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L619-L620) |
| `hidden_state` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | 레이어 입력·출력, 잔차 누적 | [`src/main.cpp:622-623`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L622-L623) |
| `rms_norms` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | RMSNorm 출력 | [`src/main.cpp:625-626`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L625-L626) |
| `buf_2048_1` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | `q_proj`·`attn_scores_v` 공유 | [`src/main.cpp:628-629`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L628-L629) |
| `k_proj_temp_buf` | `MAX_PROMPT_LEN × KV_DIM × 2` | 512 KiB | prefill K 투영 임시 버퍼 | [`src/main.cpp:635-636`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L635-L636) |
| `v_proj_temp_buf` | `MAX_PROMPT_LEN × KV_DIM × 2` | 512 KiB | prefill V 투영 임시 버퍼 | [`src/main.cpp:638-639`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L638-L639) |
| `prefill_attn_scores` | `MAX_PROMPT_LEN² × NUM_Q_HEADS × 2` | 16 MiB | prefill 어텐션 점수 | [`src/main.cpp:647-648`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L647-L648) |
| `buf_2048_2` | `MAX_BUFFER_SIZE × EMBEDDING_LENGTH × 2` | 2 MiB | `o_proj`·`down` 공유 | [`src/main.cpp:656-657`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L656-L657) |
| `gate` | `MAX_BUFFER_SIZE × HIDDEN_DIM × 2` | 8 MiB | SwiGLU gate 출력 | [`src/main.cpp:662-663`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L662-L663) |
| `up` | `MAX_BUFFER_SIZE × HIDDEN_DIM × 2` | 8 MiB | SwiGLU up 출력 | [`src/main.cpp:667-668`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L667-L668) |
| `embed_proj` | `MAX_BUFFER_SIZE × VOCAB_SIZE × 2` | 125.25 MiB | 로짓 | [`src/main.cpp:676-677`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L676-L677) |
| `embed_proj_cpu` | `MAX_BUFFER_SIZE × VOCAB_SIZE` 요소 (호스트) | 125.25 MiB | 로짓 D2H 수신 | [`src/main.cpp:681-682`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L681-L682) |
| `gpu_last_tokens` | `BATCH_SIZE × sizeof(int)` | 8 B | decode 직전 토큰 | [`src/main.cpp:684-685`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L684-L685) |
| `k_proj_batched_buffer` | `BATCH_SIZE × KV_DIM × 2` | 2 KiB | decode K 배치 | [`src/main.cpp:689-690`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L689-L690) |
| `v_proj_batched_buffer` | `BATCH_SIZE × KV_DIM × 2` | 2 KiB | decode V 배치 | [`src/main.cpp:692-693`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L692-L693) |

표의 바이트는 상수로부터 계산한 값이다. 몇 개만 직접 해 보자. `prefill_attn_scores` 는 512 × 512 × 32 = 8,388,608 요소이고 bf16 이라 16,777,216 바이트, 16 MiB 다. `gate`·`up` 은 각각 512 × 8192 = 4,194,304 요소로 8 MiB, `embed_proj` 는 512 × 128,256 = 65,667,072 요소로 125.25 MiB 다. 어휘집 `VOCAB_SIZE = 128256` 이 로짓 버퍼 크기를 지배한다. 그리고 `embed_proj` 는 배치 크기가 아니라 `MAX_BUFFER_SIZE`(512)로 잡히니, 프롬프트 길이가 이 버퍼를 정한다. 반대로 `hidden_state`·`rms_norms`·`buf_2048_1`·`buf_2048_2` 네 공용 버퍼와 `input_embeddings` 가 모두 2 MiB 로 같은 건, `MAX_PROMPT_LEN` 과 `MAX_BUFFER_SIZE` 의 값이 같아서다.

로짓 버퍼만 호스트 쪽 짝 `embed_proj_cpu` 가 있다. 이유는 argmax 가 CPU 에서 돌기 때문이다. D2H 로 내려받은 뒤, 마지막 토큰의 행에서 가장 큰 값의 인덱스를 찾는다.

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

`VOCAB_SIZE` 전체를 도는 CPU 루프가 매 토큰마다 도는 건 비싸다. 코드도 그걸 알고 있다 — "write a proper kernel for it" TODO 가 붙어 있다. argmax 를 GPU 로 옮기면 이 D2H 복사와 CPU 벡터가 통째로 사라진다.

이 구간 뒤쪽 세 버퍼는 decode 전용이다. `gpu_last_tokens` 는 decode 직전 토큰 ID 를 담고, `k_proj_batched_buffer`·`v_proj_batched_buffer` 는 슬롯별 K/V 투영을 한 번에 모으는 임시 버퍼다. 크기가 `BATCH_SIZE × KV_DIM` 이라 2 KiB 로 아주 작다.

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

prefill 의 K/V 임시 버퍼(`k_proj_temp_buf`·`v_proj_temp_buf`, 512 KiB)와 decode 의 배치 버퍼(2 KiB)가 따로인 건 두 경로가 다루는 모양이 다르기 때문이다. prefill 은 프롬프트 전체를 한 번에, decode 는 배치 슬롯마다 토큰 하나를 다룬다. 그 대비는 5편에서 본다.

## 같은 버퍼, 두 역할 — 별칭 설계

표에서 "공유"라고 적힌 두 버퍼를 보자. `buf_2048_1` 은 q 프로젝션과 어텐션 점수×V 결과가, `buf_2048_2` 는 O 프로젝션과 SwiGLU down 프로젝션 결과가 같은 메모리를 쓴다. GPU 메모리 아끼기의 전형적인 수법이다 — 서로 다른 시점에 쓰이니 한 덩어리면 충분하다.

prefill 안에서 어떤 순서로 다시 쓰이는지 보자.

```cpp
q_proj = buf_2048_1;
...
attn_scores_v = buf_2048_1;
```
— [`src/main.cpp:177`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L177), [`src/main.cpp:343`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L343)

처음에는 Q 투영이 `buf_2048_1` 에 결과를 쓴다. 그다음 헤드별 어텐션 점수 계산([`src/main.cpp:301-327`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L301-L327))이 이 Q 를 전부 읽는다. 그 뒤에야 343 에서 같은 버퍼가 점수×V 의 출력으로 다시 쓰인다. Q 가 살아 있는 동안 덮어쓰지 않으니 안전하다. 이 안전성은 소스에 나타난 "읽는 순서"라는 정적 사실로만 보장되고, 실행으로 검증된 것은 아니다.

`buf_2048_2` 도 같다.

```cpp
o_proj = buf_2048_2;
...
down = buf_2048_2;
```
— [`src/main.cpp:377`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L377), [`src/main.cpp:470`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L470)

377 에서 `o_proj = buf_2048_2` 로 다시 쓰이고, 이어지는 cuBLAS([`src/main.cpp:378-396`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L378-L396))가 O 투영 출력을 기록한다. `residualAdd`([`src/main.cpp:399`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L399))가 그 값을 읽어 `hidden_state` 에 더한 뒤, 같은 버퍼가 470 에서 down 프로젝션 출력으로 다시 쓰인다. O 와 down 의 결과가 살아 있는 시점은 겹치지 않는다.

decode 경로에서도 이 별칭이 이어진다. `pagedAttention` 의 출력이 다시 `buf_2048_1`([`src/main.cpp:878`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L878))에 기록되고, 그것이 곧바로 O 투영의 입력이 된다. 이 in-place 쓰기가 안전한 이유는 6편이 다룬다.

## 초기화 구간에 함께 준비되는 것들

계산 버퍼 16개 외에도 같은 초기화 구간에 세 가지가 만들어진다.

- paged KV cache 할당자([`src/main.cpp:574-581`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L574-L581)) — 2GB `kv_cache` 와 `free_blocks`(0..65535), `block_table`(전부 -1)이다. PagedAttention 은 KV cache 를 연속된 큰 덩어리가 아니라 작은 블록(페이지)으로 쪼개 관리하는 기법이다. 1편에서 그 준비를 보았고, 이 구조를 읽고 쓰는 어텐션 커널은 6편이 다룬다. ([Kwon et al., SOSP 2023](https://arxiv.org/pdf/2309.06180))
- RoPE 주파수 테이블([`src/main.cpp:572`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L572)) — 3편의 몫이다.
- 요청 큐([`src/main.cpp:583-594`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L583-L594)) — 7편의 몫이다.

이 편에서 함께 보는 건 슬롯·배치 상태다.

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
— [`src/main.cpp:597-610`](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/src/main.cpp#L597-L610) (일부 발췌)

`is_slot_free` 는 슬롯이 비었는지, `current_prompt_len` 은 슬롯의 현재 길이를 기억한다. `gpu_active_slots`·`gpu_seq_lens` 는 decode 루프에서 활성 슬롯 목록과 길이를 커널로 전달하는 연속 메모리다. 슬롯이 언제 비고 채워지는지는 7편의 몫이다.

## 더 읽을거리

- [README](https://github.com/jmaczan/tiny-vllm/blob/e25bf1994efa90bc98b721ba7c527402f86fbeaf/README.md) — "How floating-point numbers work and why we use bfloat16" 절이 이 편의 bf16 단위를, "GPU and CPU memory" 절이 왜 버퍼를 GPU 에 미리 잡아 두는지를 설명한다.
- [safetensors 문서 (Hugging Face)](https://huggingface.co/docs/safetensors/en/index) — 헤더 JSON + 바이트 덩어리 구조
- [PagedAttention 논문 (Kwon et al., SOSP 2023)](https://arxiv.org/pdf/2309.06180) — 2GB 캐시를 페이지로 쪼개는 배경 (6편의 배경)

## 이 글의 한계

이 시리즈는 NVIDIA GPU 와 CUDA 툴체인이 없는 환경에서 쓰여, `cudaMalloc`·`cudaMemcpy` 를 비롯해 어떤 것도 실행하지 않았다. `loadWeights` 의 동작과 표의 바이트 수는 전부 소스 인용이고, 바이트 수는 컴파일 타임 상수로부터 계산한 값이지 측정값이 아니다. 실제 `model.safetensors` 는 gated 모델 `meta-llama/Llama-3.2-1B-Instruct` 의 가중치라 확보 자체가 제약이고 적재도 검증되지 않았다. 텐서 이름이 문자열로 고정돼 있어 다른 모델 구조에는 매핑이 어긋난다. 버퍼 별칭의 안전성은 읽기 순서라는 정적 주장일 뿐 동시성·경합에 대한 실행 검증은 없다. 상수가 바뀌면(`MAX_SEQ_LEN`, `BATCH_SIZE`, `MAX_PROMPT_LEN` 등) 표의 크기가 전부 따라 바뀐다.
