#!/bin/bash

make clean
rm qlens.cpython*
make -j6 && \
    python3 test_imgdata.py && \
    echo -e "\n\nTesting Lens:" && \
    python3 test_lens.py

sleep 1
echo
echo
cat *.h | grep '^[[:blank:]]*//' | grep -e "@EXCEL"  >/dev/null

if [ $? -eq 1 ]; then
    echo -e "There are traces of debugging messages present on some headers. Double check the source code before pushing the build."
fi 

cat *.cpp | grep '^[[:blank:]]*//' | grep -e "@EXCEL"  >/dev/null

if [ $? -eq 0 ]; then
    echo -e "There are traces of debugging messages present on some C++ files. Double check the source code before pushing the build."
fi
