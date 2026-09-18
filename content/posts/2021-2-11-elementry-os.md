---
categories:
- linux
date: '2021-02-11'
description: libinput-gestures 를 설치하고 ~/.config/libinput-gestures.conf 에 3·4 손가락 스와이프를 매핑해, elementary OS 트랙패드를 맥처럼 쓰는 설정입니다.
slug: elementryos-mouch-pad-using-it-like-a-mac-touch-gestures-lokijuno
title: elementary OS 에서 맥처럼 트랙패드 제스처 쓰기 (Loki, Juno)
---

## Adding Gestures

## Add yourself to the input group 
```
sudo gpasswd -a $USER input 
```
## Install dependencies sudo apt-get install xdotool wmctrl libinput-tools and Clone and install 

```
git clone http://github.com/bulletmark/libinput-gestures 
cd libinput-gestures 
sudo ./libinput-gestures-setup install
```
## Create a custom configuration file
```
vim ~/.config/libinput-gestures.conf
```

```
gesture swipe up 4 xdotool key super+Up
gesture swipe down 4 xdotool key super+Down
gesture swipe left 4 xdotool key super+Right
gesture swipe right 4 xdotool key super+Left

gesture swipe left 3 xdotool key alt+Left
gesture swipe right 3 xdotool key alt+Right
gesture swipe up 3 xdotool key ctrl+Page_Down
gesture swipe down 3 xdotool key ctrl+Page_Up
```