# Tokens

TokenMakers consists of Tokenizer, Indexer, Vocabulary, and Embedding Modules.  
`TokenMaker` is responsible for the indexing of text and the generation of the tensors through the embedding module.


## Tokenizers

- Tokenizer Design

![images](../../images/tokenizers_design.png)

```
class SentTokenizer(name, config): ...
class WordTokenizer(name, sent_tokenizer, config) ...
class SubwordTokenizer(name, word_tokenizer, config) ...
class CharTokenizer(name, word_tokenizer, config) ...
```

The Tokenizer has a dependency with the other unit's tokenizer and the `tokenize()` function takes the input of text units.  
(* unit: unit of input e.g. 'text', 'sentence' and 'word')

- `tokenizer()` example

```
>>> text = "Hello World.This is tokenizer example code."
>>> word_tokenizer.tokenize(text, unit="text")  # text -> sentences -> words
>>> ['Hello', 'World', '.', 'This', 'is', 'tokenizer', 'example', 'code', '.']
>>> word_tokenizer.tokenize(text, unit="sentence")  # text -> words
>>> ['Hello', 'World.This', 'is', 'tokenizer', 'example', 'code', '.']
```

Several tensors in a sub-level text unit can be combined into a single tensor of higher level via a vector operation. For example, subword level tensors can be averaged to represent a word level tensor.

e.g.) concatenate \[word; subword\] (subword tokens --average--> word token) 


* The list of pre-defined `Tokenizers`:

| Text Unit | Language | Name | Example |
| ---- | ---- | --- | --- |
| Char | All | **character** | Hello World<br/>-> ["Hello", "World"]<br/>-> [["H", "e", "l", "l", "o"], ["W", "o", "r", "l", "d"]] |
| Char | Korean | [**jamo_ko**](https://github.com/rhobot/Hangulpy) | "안녕 세상"<br/>-> ["안녕", "세상"]<br/>-> [["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"], ["ㅅ", "ㅔ", "ㅅ", "ㅏ", "ㅇ"]] |
| Subword | All (but, need vocab.txt) | [**wordpiece**](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py) | "expectancy of anyone"<br/>-> ["expectancy", "of", "anyone"]<br/>-> ["expect", "##ancy", "of", "anyone"] |
| Word | English | [**nltk_en**](http://www.nltk.org/api/nltk.tokenize.html) | - |
| Word | English | [**spacy_en**](https://spacy.io/api/tokenizer) | - |
| Word | Korean | [**mecab_ko**](https://bitbucket.org/eunjeon/mecab-ko) | - |
| Word | All | **bert_basic** | - |
| Word | All | **space_all** | "Hello World"<br/>-> ["Hello", "World"] |
| Sent | All | [**punkt**](http://www.nltk.org/api/nltk.tokenize.html) | "Hello World. This is punkt tokenizer."<br/>-> ["Hello World.", "This is punkt tokenizer."] |


## Token Maker

* The list of pre-defined `Token Maker`:

| Type | Description | Category | Notes |
| ---- | ---- | --- | --- |
| **char** | character -> convolution -> maxpool | `CharCNN` | - |
| **cove** | Embeddings from Neural Machine Translation | `NMT` | - From [Salesforce](https://github.com/salesforce/cove) |
| **word** | word -> Embedding (+pretrained) | `Word2Vec` | - |
| **frequent_word** | word token + pre-trained word embeddings fixed and only fine-tune the N most frequent | `Word2Vec` + `Fine-tune` | - |
| **exact_match** | Three simple binary features, indicating whether p_i can be exactly matched to one question word in q, either in its original, lowercase or lemma form. | `Feature` | - Sparse or Embedding<br/> - Only for RC|
| **elmo** | Embeddings from Language Models | `LM` | From [Allennlp](https://github.com/allenai/allennlp) |
| **linguistic** | Linguistic Features like POS Tagging, NER and Dependency Parser | `Feature` | - Sparse or Embedding |


- Example of tokens in [BaseConfig](#baseconfig)

```
"token": {
   "names": ["char", "glove"],
   "types": ["char", "word"],
   "tokenizer": {  # Define the tokenizer in each unit.
       "char": {
           "name": "character"
       },
       "word": {
           "name": "treebank_en",
           "split_with_regex": true
       }
   },
   "char": {  # token_name
       "vocab": {
           "start_token": "<s>",
           "end_token": "</s>",
           "max_vocab_size": 260
       },
       "indexer": {
           "insert_char_start": true,
           "insert_char_end": true
       },
       "embedding": {
           "embed_dim": 16,
           "kernel_sizes": [5],
           "num_filter": 100,
           "activation": "relu",
           "dropout": 0.2
       }
   },
   "glove": {  # token_name
       "indexer": {
           "lowercase": true
       },
       "embedding": {
           "embed_dim": 100,
           "pretrained_path": "<glove.6B.100d path>,
           "trainable": false,
           "dropout": 0.2
       }
   }
},

# Tokens process
#   Text -> Indexed Featrues -> Tensor -> TokenEmbedder -> Model

# Visualization
# - Text: Hello World
# - Indexed Feature: {'char': [[2, 3, 4, 4, 5], [6, 7, 8, 4, 9]], 'glove': [2, 3]} 
# - Tensor: {'char': tensor, 'glove': tensor} 
# - TokenEmbedder: [char; glove]  (default: concatenate)
# - Model: use embedded_value
```