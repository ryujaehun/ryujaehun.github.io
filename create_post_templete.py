#!/usr/bin/env python3

import os
import datetime
import requests
import argparse
import pwd

def get_arxiv_info(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        # 제목 추출
        title_start = content.find('<title>') + len('<title>')
        title_end = content.find('</title>', title_start)
        title = content[title_start:title_end].strip()
        # 첫 번째 라인은 일반적으로 'arXiv:ID'이므로 제외
        title_lines = title.split('\n')
        if len(title_lines) > 1:
            title = title_lines[1].strip()
        return title
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate Jekyll blog posts from arXiv PDFs.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing PDF files.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where MD files will be saved.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # 현재 날짜 가져오기
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    # 'jaehun'의 UID와 GID 가져오기
    user_name = 'jaehun'
    try:
        pw_record = pwd.getpwnam(user_name)
        uid = pw_record.pw_uid
        gid = pw_record.pw_gid
    except KeyError:
        print(f"User '{user_name}' not found.")
        return

    # 입력 폴더의 모든 PDF 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            # 파일명에서 arXiv ID 추출
            arxiv_id = filename[:-4]

            # arXiv 정보 가져오기
            title = get_arxiv_info(arxiv_id)
            if title:
                # 파일명 생성
                md_filename = f"{today}-paper-{arxiv_id}.md"

                # 마크다운 파일 내용 작성
                md_content = f"""---
title: "{title}"
date: {today}
categories:
 - paper-review
 - with-gpt
tags:
 - {arxiv_id}
 - 
---

[논문 링크](https://arxiv.org/abs/{arxiv_id})

# 공통질문
## master prompt

이 논문을 자주 자세하게 읽고 논문에 대한 자세한 리뷰를 작성하기 위해서 질의 응답을 한 후 이를 바탕으로 블로그 포스트를 작성할 것이다 이러한 목적과 github를 blog jekyll-theme-next 테마를 사용하고 있다는 점을 염두해서 markdown 형식으로 작성한다.

# 프롬프트 1.1.1 (연구의 공백): "논문의 'Introduction'과 'Related Work' 섹션을 분석하여, 이 연구가 명시적으로 해결하고자 하는 핵심적인 연구 공백(research gap), 기존 연구의 결정적 한계, 또는 미해결 질문이 무엇인지 설명해 줘. 저자들이 설명하는, 이 논문 출판 시점의 '최신 기술(state of the art)'은 어떤 상태였는지 요약해 줘."

# 프롬프트 1.1.2 (핵심 가설): "이 논문의 중심 가설(central hypothesis) 또는 핵심 주장은 무엇인가? '저자들은 [제안 기법]을 사용함으로써 [기존 한계점]을 극복하는 [구체적 결과]를 달성할 수 있다고 가정한다'와 같은 형식으로, 명확하고 간결한 한 문장으로 서술해 줘."

# 프롬프트 1.2.1 (독창성 식별): "논문 전체를 바탕으로, 가장 중요하고 독창적인 기여(contribution) 1~3가지를 구별되는 항목으로 나열해 줘. 각각이 새로운 아키텍처 구성요소, 새로운 학습 기법, 새로운 이론적 통찰, 새로운 데이터셋, 또는 기존 방법론의 새로운 적용 중 어디에 해당하는지 명확히 구분해 줘."

# 프롬프트 1.2.2 (저자 관점에서의 강점): "저자들의 관점에서, 자신들의 접근법이 이전 방법들보다 우월한 이유는 무엇인가? 그들이 자신들의 연구가 지닌 독창성과 강점을 뒷받침하기 위해 사용하는 핵심 논거를 인용하거나 알기 쉽게 설명해 줘."

# 프롬프트 1.3.1 (알고리즘 단계별 설명): "핵심 알고리즘, 모델 아키텍처, 또는 주요 방법론을 단계별(step-by-step)로 설명해 줘. 독자는 AI 분야의 대학원생 수준이라고 가정해. 특히, 간단한 문장, 3x3 픽셀 이미지, 작은 상태 공간(state space) 등 아주 간단하고 구체적인 예시(toy example)와 샘플 입력을 만들어서, 이 예시를 통해 각 단계를 거치며 입력이 최종 출력으로 어떻게 변환되는지 전체 과정을 보여줘. 등장하는 모든 핵심 용어와 변수는 그 즉시 정의해 줘."

# 프롬프트 1.3.2 ('비밀 병기' 식별): "이 논문의 핵심 기여를 가능하게 하는 가장 결정적인 단일 수학 공식, 알고리즘 단계, 또는 아키텍처 구성요소를 식별해 줘. 그것의 기능과, 그것이 이 방법론의 성공에 왜 필수적인지 설명해 줘."

# 프롬프트 1.4.1 (핵심 결과 분석): "'Experiments' 또는 'Results' 섹션의 표와 그림을 포함한 주요 결과를 분석해 줘. 사용된 핵심 성능 지표(performance metrics)는 무엇인가? 어떤 벤치마크 데이터셋에서 결과가 보고되었는가? 저자들이 자신들의 방법론의 성공 증거로 가장 강조하는 주요 결과를 요약해 줘."

# 프롬프트 1.4.2 (비판적 비교): "제안된 방법론은 논문에서 언급된 주요 베이스라인 및 SOTA(State-of-the-Art) 모델들과 비교하여 어떤 성능을 보이는가? 저자들의 우월성 주장을 가장 강력하게 뒷받침하는 특정 결과나 비교 지점을 식별해 줘. 반대로, 제안된 방법론이 경쟁 모델을 능가하지 못했거나 개선 효과가 미미했던 결과는 없는지 찾아봐. 만약 있다면, 저자들은 이러한 경우에 대해 어떤 이유를 제시하는가?"

# 프롬프트 1.5.1 (언급된 한계와 잠재적 한계): "저자들이 논문에서 명시적으로 인정한 한계점, 약점, 또는 실패 사례(failure modes)는 무엇인가? 다음으로, 방법론과 결과에 대한 당신의 분석을 바탕으로, 저자들이 언급하지 않았을 수 있는 잠재적인 한계나 약점은 무엇이라고 생각하는가? (예: 강력한 가정에 대한 의존성, 확장성 문제, 높은 연산 비용, 일반화의 한계, 잠재적인 부정적 사회 영향 등)"

# 프롬프트 1.5.2 (미래 연구 궤적): "저자들이 제안하는 구체적인 향후 연구 방향은 무엇인가? 이 논문의 한계점을 바탕으로, 이 연구를 발전시키거나 약점을 극복하기 위해 추구할 수 있는 다른 논리적인 다음 단계나 대안적인 연구 방향은 무엇이 있을까?"

# 주제별 추가 질문

## 모듈 A: 컴퓨터 비전 (cs.CV) 논문용

# 데이터 및 전처리: "학습 및 추론에 사용된 이미지 해상도는 얼마인가? 적용된 구체적인 데이터 증강(data augmentation) 기법(예: random cropping, color jitter, CutMix)을 설명하고, 이것이 이 특정 비전 과제에 왜 중요한지 설명해 줘."

# 모델 아키텍처: "사용된 백본 아키텍처는 무엇인가(예: ResNet, ViT, ConvNeXt)? 공간적 특징(spatial features)은 어떻게 추출되고 융합되는가? 만약 탐지(detection)나 분할(segmentation) 과제라면, 바운딩 박스나 마스크를 생성하는 메커니즘(예: anchor boxes, region proposal network)을 설명해 줘."

# 평가 및 지표: "정확도(accuracy) 외에 어떤 다른 지표가 사용되었는가(예: 탐지를 위한 mAP, 분할을 위한 IoU)? 시각적 결과에 대한 정성적 분석이 있는가? 있다면, 모델이 인상적으로 성공한 예시와 실패한 예시를 하나씩 설명해 줘."

## 모듈 B: 자연어 처리 (cs.CL) 논문용

# 데이터 및 전처리: "어떤 토큰화(tokenization) 전략이 사용되었는가(예: BPE, WordPiece, SentencePiece)? 어휘에 없는 단어(Out-of-vocabulary words)는 어떻게 처리되는가? 텍스트 정제 및 정규화 단계를 설명해 줘."

# 모델 아키텍처: "트랜스포머를 사용했다면, 어텐션 메커니즘의 구성(예: 헤드 수, 레이어 수)을 상세히 설명해 줘. 위치 인코딩(positional encodings)은 어떻게 처리되는가? Seq2Seq 모델이라면, 인코더-디코더 상호작용을 설명해 줘."

# 학습 및 최적화: "구체적인 언어 모델링 목표(objective)는 무엇인가(예: Causal LM, Masked LM, Prefix LM)? 모델은 어떤 코퍼스로 사전학습(pre-trained)되었는가? 다운스트림 과제를 위한 파인튜닝(fine-tuning) 전략을 설명해 줘."

## 모듈 C: 강화 학습 (cs.LG/cs.AI) 논문용

# 모델 아키텍처/알고리즘: "상태 공간(State Space), 행동 공간(Action Space), 보상 함수(Reward Function)를 정의해 줘. 모델은 온-폴리시(on-policy)인가 오프-폴리시(off-policy)인가? 가치 기반(예: DQN), 정책 기반(예: REINFORCE), 또는 액터-크리틱(예: A2C, PPO) 중 어느 유형인가? 정책 및/또는 가치 함수에 대한 핵심 업데이트 규칙을 설명해 줘."

# 학습 및 최적화: "탐험(exploration)과 활용(exploitation)의 트레이드오프는 어떻게 관리되는가(예: epsilon-greedy, entropy regularization)? 리플레이 버퍼(replay buffer)가 사용되었는가? 학습에 사용된 시뮬레이션 환경이나 실제 환경 설정을 설명해 줘."

# 평가 및 지표: "성능은 어떻게 측정되는가(예: 누적 보상, 에피소드 길이, 성공률)? 평가에 얼마나 많은 에피소드나 타임스텝이 사용되었는가? 시간에 따른 성능을 보여주는 학습 곡선(learning curves)이 있는가?"

## 모듈 D: 시스템 및 구현 (cs.DC, cs.AR 등) 논문용

# 구현 및 자원: "핵심 소프트웨어 의존성(예: CUDA, MPI, 특정 라이브러리)은 무엇인가? 학습 및 추론 중 예상되는 메모리 점유량(GPU의 경우 VRAM, CPU의 경우 RAM)은 얼마인가? 명시된 하드웨어에서의 처리량(throughput)은 얼마인가(예: images/sec, tokens/sec)? 총 연산 비용(예: 총 FLOPs 또는 Petaflop-days)에 대한 상세한 내역을 제공해 줘."

# 평가 및 지표: "평가를 위한 주요 지표는 무엇인가: 지연 시간(latency), 처리량(throughput), 전력 소비, 또는 비용 대비 성능? 시스템은 더 많은 데이터, 사용자, 또는 컴퓨팅 노드에 따라 어떻게 확장되는가?"

# 정리를 위한 마스터 프롬프트

"당신은 저명한 AI 블로그의 전문 AI 연구원이자 기술 작가입니다. 당신의 독자는 AI 실무자, 연구원, 학생들로 구성되어 있습니다. 지금부터 한 연구 논문에 기반한 일련의 질문-답변 쌍을 제공하겠습니다. 당신의 임무는 이 모든 정보를 하나의 일관되고 잘 구조화된 블로그 포스트로 종합하는 것입니다. 포스트는 반드시 마크다운으로 작성해야 합니다.
아래의 구조를 정확히 따르세요:


한 줄 요약 (TL;DR)
핵심 아이디어
배경: 그들이 해결한 문제
새로운 접근법: Method Name
작동 원리: 구체적인 예시로 살펴보기
성능 검증: 주요 결과
우리의 관점: 강점, 한계, 그리고 이 연구가 중요한 이유
다음 단계는?: 앞으로의 길
비판적이면서도 공정한 어조를 유지하세요. 복잡한 개념을 명확하고 간결하게 설명하세요. 섹션 간의 전환이 자연스럽도록 하세요. 단순히 답변을 복사-붙여넣기 하지 말고, 매력적인 서사로 엮어내세요.

[여기에 이전의 모든 Q&A 쌍을 붙여넣기]"



"""

                # 마크다운 파일 생성
                md_file_path = os.path.join(output_folder, md_filename)
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                print(f"Created {md_filename} in {output_folder}")

                # 파일 소유권 변경
                os.chown(md_file_path, uid, gid)

            else:
                print(f"Failed to retrieve information for arXiv ID {arxiv_id}")

if __name__ == '__main__':
    main()