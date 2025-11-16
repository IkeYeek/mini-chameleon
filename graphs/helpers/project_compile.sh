guix shell --pure -D mini-chameleon bash gcc-toolchain -- bash --norc <<'EOF'
    mkdir -p ../build/debug
    cmake --build ../build/debug
EOF