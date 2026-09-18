---
categories:
- linux
date: '2021-02-11'
description: python3 로 띄웠는데 python2 커널이 실행될 때 `ipython3 kernel install` 로 커널을 다시 등록하는 방법입니다.
slug: jupyter-notebook-다른python이-실행될-시
title: Jupyter Notebook 에서 python2 커널이 잡힐 때
---

jupyter notebook에서 python3 를 실행하였는데 python2 커널이 계속 실행되었는데 아래외 같이 해결하면 된다.
```
ipython3 kernel install
```
references

https://github.com/jupyter/jupyter/issues/270