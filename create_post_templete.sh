#!/bin/bash

# 기본 생성 개수
NUM_POSTS=${1:-5}

# 현재 날짜를 한국 시간으로 설정
DATE=$(TZ=Asia/Seoul date +"%Y-%m-%d")

# _posts 폴더가 없으면 생성
mkdir -p _posts

# 파일 생성
for ((i=1; i<=NUM_POSTS; i++)); do
    FILENAME="_posts/${DATE}-paper-name${i}.md"
    cat <<EOF > "$FILENAME"
---
title: 
categories:
 - paper-review
 - with-gpt
tags:
---

논문 : 

아래 글은 Scholar GPT와 대화를 나눈 내용입니다.

# Q : 논문의 핵심 내용과 강점, 알고리즘 설명, 그리고 한계점

# A : 
EOF
    echo "생성된 파일: $FILENAME"
done

echo "$NUM_POSTS개의 파일이 생성되었습니다."
