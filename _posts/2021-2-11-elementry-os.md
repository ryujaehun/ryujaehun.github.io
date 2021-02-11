---
title: ElementryOS mouch pad Using it like a Mac Touch Gestures (Loki,Juno)
categories:
 - linux
tags:
---

# Adding Gestures

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