mkdir -p data/wikisql
cd data/wikisql
wget https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2
tar xvjf data.tar.bz2

mv data/* .

rm data.tar.bz2
rm -r data
