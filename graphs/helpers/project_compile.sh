guix shell --pure -D mini-chameleon bash gcc-toolchain -- bash --norc <<'EOF'
    mkdir -p ../build/debug
    cmake ../ -B ../build/debug/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_STARPU=ON # -DENABLE_CUDA=ON
    cmake --build ../build/debug
EOF