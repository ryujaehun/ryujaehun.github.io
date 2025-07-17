---
categories:
- compiler
date: "2021-02-12T00:00:00Z"
tags: null
title: LLVM (clang) build and install (ubuntu 18.04)
---
# clone llvm repo
```
git clone -b llvmorg-10.0.0 https://github.com/llvm/llvm-project.git llvm10
```
# configure
Ninja를 사용하면 컴파일 시간을 많이 단축할 수 있다.
```
cd llvm9
mkdir build
cmake -DLLVM_ENABLE_PROJECTS="clang;"   -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON -G "Ninja" ../llvm
ninja
```

pass를 만들다보면 자주 컴파일할 상황이 생기기 때문에 ccache와 ninja를 사용해서 빌드 속도를 높혀주는것을 추천한다.(-LLVM_CCACHE_BUILD ON)
만약에 ninja가 없을시에는 -G “Unix Makefile”로 바꾸자
위 설정에서 build type이 Debug일 시에 linking시 많은 ram이 소요되니 주의 바란다.
위에서 빌드한 clang을 특정 위치에 설치하고 싶다면 -DCMAKE_INSTALL_PREFIX 옵션을 사용하자.