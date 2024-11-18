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
---

[논문 링크](https://arxiv.org/abs/{arxiv_id})

# Q : 이 논문을 아주 자세하게 읽고 논문의 강점과 독창적인 지점을 설명해주고 핵심 알고리즘을 예시 입력을 들어서 전체적인 과정을 설명해줘 추가적으로 논문의 한계점에 대해서도 알려줘

# A :

# Q : 이 논문에서 사용하는 방법을 학습하기 위하여 어떤 데이터셋이 필요할까? 그리고 어떻게 학습을 진행하면 될지 예시를 들어서 아주 자세하게 설명해줘

# A :

# Q : 이 논문에서 제시한 결과를 자세하게 보고 다른 방법론에 비하여 특출난 점과 논문에서 제기하는 어떠한 방법이 이러한 결과를 도출하게 되었는지 논문에서 제시하는 이유와 너의 생각을 알려줘

# A :

# Q : 이 논문에서 제시된 방법을 실제로 사용하려면 기존 방법에서 어떠한 구현이 추가적으로 필요하고 이러한 구현에 소모되는 공수 및 연산에 필요한 컴퓨팅 자원의 크기에 대해서 계산해줄 수 있겠니? 

# A :

# Q : 이 논문의 입력데이터와 추론 과정에 대해서 예시를 들어 아주 자세하게 설명해 주겠니? 추가적으로 모델아키텍처의 구성 및 모델을 구성하는 연산과 메모리 요구량 컴퓨팅 요구량 대해서도 설명해줘

# A :

# Q : 이 논문의 한계를 극복하기 위한 방법으로 어떤 연구흐름이 있는지 정리해서 자세하게 설명해 줘

# A :
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