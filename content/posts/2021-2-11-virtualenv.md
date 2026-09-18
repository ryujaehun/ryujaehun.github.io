---
categories:
- linux
date: '2021-02-11'
description: virtualenv 설치부터 가상환경 생성·활성화·비활성화까지, 파이썬 버전을 지정해 환경을 격리하는 기본 사용법입니다.
slug: virtualenv로-파이썬-환경-격리하기
title: virtualenv로 파이썬 환경 격리하기
---

## About
가끔 파이썬 환경을 격리 할 필요가 있다. 이런 상황에서 virtualenv는 큰 도움이 된다.

## Install
```
sudo pip install virtualenv
```
## Usage

__가상환경 생성__
```
virtualenv -p [python 버전 ex)python3.6] [가상환경 이름 ex)env]

```

__가상환경 활성화(활성화 후 bash에 (가상환경 이름)가 prefix로 붙는 것을 볼 수 있다.)__
```
source ./가상환경 이름/bin/activate
```

__가상환경 비활성화__
```
deactivate
```