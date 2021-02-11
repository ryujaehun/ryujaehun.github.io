---
title: Ubuntu pdf 를 이미지로 변환
categories:
 - linux
tags:
---
## About
Pdf 를 이미지로 변환해야할 일이 종종 있어 구글에서 검색되는 사이트를 이용하곤 하지만 많은 양의 pdf 를 변환해야 하거나 개인정보가 있는 데이터를 서버에 올리기 꺼림직 한 경우가 있다 이럴때 터미널에서 직접 pdf 를 변환해보자.
pdftoppm 를 이용할 것이며 다음과 같이 변환 하면 된다.
## Example
```
pdftoppm filename.pdf output_name options
```

## Option
```
-r number
          Specifies the X and Y resolution, in DPI.  The  default  is  150
          DPI.
-gray  Generate a grayscale PGM file (instead of a color PPM file).

-png   Generates a PNG file instead a PPM file.

-jpeg  Generates a JPEG file instead a PPM file.

-jpegopt jpeg-options
       When  used  with  -jpeg,  takes a list of options to control the
       jpeg compression. See JPEG OPTIONS for the available options.

-tiff  Generates a TIFF file instead a PPM file.

-tiffcompression none | packbits | jpeg | lzw | deflate
       Specifies the TIFF compression type.  This defaults to "none".

```