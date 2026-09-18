---
categories:
- linux
date: '2021-02-11'
description: '`docker search` 는 이미지만 찾아 주고 태그는 보여 주지 않습니다. 레지스트리 API 를 긁어 태그 목록을 뽑는 셸 함수를 ~/.zshrc 에 넣어 씁니다.'
slug: docker-tag-검색하기
title: docker 이미지의 태그 목록 검색하기
---

도커에서 이미지를 검색할시는 아래처럼 도커의 내장명령어를 사용하면 된다 
```
docker search images
```
그런데 tag검색은 당연히 있을줄 알았는데 존재하지 않아서 당황스러웠다. 다행히도 찾는 방법이 존재하였다. 

아래함수를 ~/.zshrc에 추가한다. (bash사용시~/.bashrc)

```
#usage list-dh-tags <repo>
#example: docker-tag node
function docker-tag(){
    wget -q https://registry.hub.docker.com/v1/repositories/$1/tags -O -  | sed -e 's/[][]//g' -e 's/"//g' -e 's/ //g' | tr '}' '\n'  | awk -F: '{print $3}'
}
```