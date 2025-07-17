---
categories:
- linux
date: "2021-02-11"
tags: null
title: inplace_swap
---

```
void inplace_swap(int *x,int *y)
{
 *y=*x^*y;
 *x=*x^*y;
 *y=*x^*y;
 }
```