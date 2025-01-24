#!/bin/zsh
source /srv01/agrp/yuvalzu.zshrc
topdir="/srv01/agrp/yuvalzu/mattiasdata"
cd ${topdir}
lsetup "root 6.20.06-x86_64-centos7-gcc8-opt"

EXE="/usr/local/anaconda/3.8/bin/python3 scripts//dothestuff_divide.py "

echo "++ Launch "${EXE}
${EXE}
