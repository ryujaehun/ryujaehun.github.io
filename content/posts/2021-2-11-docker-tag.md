---
categories:
- linux
date: "2021-02-11"
tags: null
title: docker tag 검색하기
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