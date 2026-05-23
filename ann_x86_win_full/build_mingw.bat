@echo off
setlocal

if not exist build mkdir build
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo Build complete: build\ann_x86_win.exe
