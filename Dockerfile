FROM jmin/pytorch:apex
RUN git clone https://github.com/naver/claf && cd claf && pip install -r requirements.txt && python setup.py install

RUN apt-get install g++ default-jdk
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

RUN python -m nltk.downloader punkt --dir /usr/share/nltk_data
RUN python -m nltk.downloader wordnet  --dir /usr/share/nltk_data
