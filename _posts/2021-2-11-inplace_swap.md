---
title: inplace_swap
categories:
 - linux
tags:
---

```
void inplace_swap(int *x,int *y)
{
 *y=*x^*y;
 *x=*x^*y;
 *y=*x^*y;
 }
```