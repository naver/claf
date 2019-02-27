#!/bin/sh

OUT_DIR="${1:-./mecab}"

mkdir -v -p $OUT_DIR

apt-get install git cmake make automake wget

wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz

mv mecab-0.996-ko-0.9.2.tar.gz "$OUT_DIR/"
mv mecab-ko-dic-2.1.1-20180720.tar.gz "$OUT_DIR/"

cd "$OUT_DIR"

tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2.tar.gz
./configure
make
make check
sudo make install
cd ../

sudo ldconfig
tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720.tar.gz
./configure
make
sudo make install

cd ../
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996
python3 setup.py build
sudo python3 setup.py install
