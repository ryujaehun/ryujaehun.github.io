---
categories:
- python
date: "2021-02-12"
tags: null
title: 자주쓰는 파이썬 스크립트 패턴
---

개인적으로 shell보다는 python를 자주 사용하기 때문에 자주 사용하는 패턴들을 간단하게 정리
## glob,os
glob는 유닉스 스타일 경로명 패턴 확장 라이브러입니다. 이것과 os 라이브러리를 이용하면 간단하게 파일을 찾거나 바꿀수 있습니다.
개인적으로 실험 결과를 파싱할때 많이 사용하는 라이브러리 입니다.

예시)result폴더에서 png파일과 npy파일이 모두 존재할 시에 npy파일을 조작
```
import numpy as np
import glob
import os

...

for name in glob.glob('result/**/*.png',recursive=True):
    np_name=name.replace('png','npy')
    if os.path.exists(np_name):
        np.load(np_name)
        ....
```

## subprocess
해당 명령어는 파이썬에서 shell명령어를 실행시키는 명령어 입니다.
개인적으로 파라미터 등을 바꾸어가면서 실험을 진행 할때 많이 사용합니다.
예시 parameter A와 B를 바꾸어 가면서 실험.
```
import subprocess
basetext='python3 text.py --numA _numA --numB _numB 2>&1'
index=0                                 
for numA in [30,40,50]:
    for numB in [1.0,1.5,2.0]:
        text=basetext
        text=text.replace('_numA',f'{numA}')
        text=text.replace('_numB',f'{numB}')
        proc = subprocess.Popen( text , shell=True, executable='/bin/bash')
        proc.communicate()
```


## schedule
파이썬의 crontab.
```
import schedule
import time

def job():
    print("Do Job...!!!")

schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at("10:30").do(job)
schedule.every(5).to(10).minutes.do(job)
schedule.every().monday.do(job)
schedule.every().wednesday.at("13:15").do(job)
schedule.every().minute.at(":17").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```