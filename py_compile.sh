#!/bin/bash

make clean
rm qlens.cpython*
make -j6 && \
    python3 test_imgdata.py && \
    echo -e "\n\nTesting Lens:" && \
    python3 test_lens.py

