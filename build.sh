#!/usr/bin/env bash
set -euo pipefail

# Simple build script using g++ (no CMake).
# You can later blank/modify parts of this for the assignment.

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -Wall -Wextra -O2"

mkdir -p bin
rm -rf logs/

echo "[build] Compiling common.cpp"
$CXX $CXXFLAGS -Iinclude -c src/common.cpp -o bin/common.o

echo "[build] Compiling math_layer.cpp"
$CXX $CXXFLAGS -Iinclude -c src/math_layer.cpp -o bin/math_layer.o

echo "[build] Compiling preprocess.cpp"
$CXX $CXXFLAGS -Iinclude src/preprocess.cpp bin/common.o bin/math_layer.o -o bin/preprocess

echo "[build] Compiling forward_layer.cpp"
$CXX $CXXFLAGS -Iinclude src/forward_layer.cpp bin/common.o bin/math_layer.o -o bin/forward_layer

echo "[build] Compiling backward_layer.cpp"
$CXX $CXXFLAGS -Iinclude src/backward_layer.cpp bin/common.o bin/math_layer.o -o bin/backward_layer

echo "[build] Compiling logger.cpp"
$CXX $CXXFLAGS -Iinclude src/logger.cpp -o bin/logger

echo "[build] Compiling trainer.cpp"
$CXX $CXXFLAGS -Iinclude src/trainer.cpp -o bin/trainer

echo "[build] Done."
