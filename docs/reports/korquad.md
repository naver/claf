# KorQuAD

`Span Detector`

- [`KorQuAD`](https://korquad.github.io/): KorQuAD는 한국어 Machine Reading Comprehension을 위해 만든 데이터셋입니다. 모든 질의에 대한 답변은 해당 Wikipedia 아티클 문단의 일부 하위 영역으로 이루어집니다. Stanford Question Answering Dataset(SQuAD) v1.0과 동일한 방식으로 구성되었습니다.
	- v1.0
		- Train: 60359 / Dev: 5774 

---

## Results

- Dev Set

| Model | EM | F1 | BaseConfig | Note |
| --- | --- | --- | --- | --- |
| **BiDAF** | 75.476 | 85.915 | korquad/bidaf.json | - |
| **DocQA** | 77.693 | 88.115 | korquad/docqa.json | - |
| **BERT**-Base, Multilingual Uncased | 77.641 | 87.851 | korquad/bert_base_multilingual_uncased.json | - |
| **BERT**-Base, Multilingual Cased | 78.957 | 88.686 | korquad/bert_base_multilingual_cased.json | - |