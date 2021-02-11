---
title: Logitech MX anywhere 2s 우분투에서 제스쳐 사용하기
categories:
 - linux
tags:
---

Logitech 마우스는 options 라는 소프트웨어를 기본적으로 제공하여 키맵변경 logitech flow 등 많은 기능을 사용할 수 있다. 그러나 options 가 mac과 windows에서만 제공하는 바람에 9만원이나 하는 마우스가 그냥 돌덩이가 되어버렸다. 그러나 역시 리눅스에서 사용할 수 있는 방법이 있었다.  

아래 방법을 사용하여 xautomation xbindkeys 을 설치 후 마우스의 키세팅을 알아본다.(버튼은 글쓴이가 다 알아놨으니 걱정하지 않아도 된다.)

## 설치

1. 배시창을 연다
1. apt -y install xautomation xbindkeys 의존성을 포함하여 설치를 한다.
1. xev | tee mouse.log 실행시 검은박스가 나타난다.
1. 마우스의 기능키를 순서대로 눌러본다.
1. 버튼의 동작이 mouse.log에 저장되는데 그것을 토대로 xbindkeysrc를 작성하게 된다. 
1. 아래는 anywhere 2s 의 키 이다. 

```
ButtonPress event, serial 40, synthetic NO, window 0x7800001,root 0x1c9, subw 0x0, time 1311199, (71,71), root:(1201,100),state 0x10, button 9, same_screen YES
```



위 처럼나올껀데 버튼의 상태값과 버튼의 값이 중요하다.
아래는 버튼 값이다. 우리는 버튼클릭만을 이용할 것이기에 0x10만 사용하면 된다.
![](../assets/images/mx-anywhere-2s-1.png)

~/.xbindkeysrc 을 편집한다. 아래는 글쓴이의 코드이다.
```

"xte 'keydown Control_L' 'keydown Alt_L'  'key Down' 'keyup Alt_L' 'keyup Control_L'"
    m:0x10 + b:6
"xte 'keydown Control_L' 'keydown Alt_L' 'key Up' 'keyup Alt_L' 'keyup Control_L'"
    m:0x10 + b:7
"xte 'keydown Control_L' 'key W' 'keyup Control_L'"
    m:0x10 + b:2
"xte 'keydown Alt_L' 'key Right' 'keyup Alt_L'"
    m:0x10 + b:9
"xte 'keydown Alt_L' 'key Left' 'keyup Alt_L'"
    m:0x10 + b:8
```



2번은 창닫기 8,9번은 뒤로가기,앞으로가기 7,6번은 워크스페이스 변경이다.


시작프로그램에 /usr/bin/xbindkeys를 추가한다.

# Reference
- https://blog.onee3.org/2016/09/how-to-get-logitech-mx-anywhere-2-to-work-with-ubuntu/