---
categories:
- linux
date: "2021-02-11T00:00:00Z"
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