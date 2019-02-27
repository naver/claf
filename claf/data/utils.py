
from collections import defaultdict

import numpy as np
import torch


def make_bert_token_types(bert_inputs, SEP_token="[SEP]"):
    """
    Bert Inputs segment_ids

    ex) [CLS] hi [SEP] he ##llo [SEP] => 0 0 0 1 1 1

    * Args:
        bert_inputs: feature dictionary consisting of
            - text: text from data_reader
            - token_name: text converted to corresponding token_type

    * Kwargs:
        SEP_token: SEP special token for BERT
    """

    feature_keys = list(bert_inputs[0].keys())  # TODO: hard-code
    if "text" in feature_keys:
        feature_keys.remove("text")

    feature_key = feature_keys[0]

    token_types = []
    for bert_input in bert_inputs:
        token_type = make_bert_token_type(bert_input["text"], SEP_token=SEP_token)
        token_types.append({feature_key: token_type})
    return token_types


def make_bert_token_type(bert_input_text, SEP_token="[SEP]"):
    SEP_index = bert_input_text.index(SEP_token) + 1

    token_type = [0] * SEP_index
    token_type += [1] * (len(bert_input_text) - SEP_index)

    assert len(token_type) == len(bert_input_text)
    return token_type


def padding_tokens(tokens, max_len=None, token_name=None, pad_value=0):
    """ Padding tokens according to token's dimension """

    def _pad_tokens(seqs, maxlen, pad_id=0):
        lens = [len(seq) for seq in seqs]

        if pad_id == 0:
            padded_seqs = torch.zeros(len(seqs), maxlen).long()
        else:
            padded_seqs = torch.ones(len(seqs), maxlen).long() * pad_id

        for i, seq in enumerate(seqs):
            if type(seq[0]) == dict:
                pass
            else:
                seq = [int(s) for s in seq]
                end = lens[i]
                padded_seqs[i, :end] = torch.LongTensor(seq)
        return padded_seqs

    def _pad_char_tokens(seqs, seq_maxlen, char_minlen=10, char_maxlen=None, pad_value=0):
        if char_maxlen is None:
            char_maxlen = max([len(chars) for seq in seqs for chars in seq])
            if char_maxlen < char_minlen:
                char_maxlen = char_minlen

        padded_chars = torch.zeros(len(seqs), seq_maxlen, char_maxlen).long()
        for i in range(len(seqs)):
            char_tokens = _pad_with_value(seqs[i], seq_maxlen, pad_value=[[pad_value]])
            padded_chars[i] = _pad_tokens(char_tokens, char_maxlen, pad_id=pad_value)
        return padded_chars

    def _pad_with_value(data, size, pad_value=[0]):
        if type(pad_value) != list:
            raise ValueError("pad_value data type is list.")

        return data + pad_value * (size - len(data))

    token_dim = get_token_dim(tokens)
    if token_dim > 1 and max_len is None:
        max_len = max(len(token) for token in tokens)

    if token_dim == 2:  # word
        return _pad_tokens(tokens, max_len, pad_id=pad_value)
    elif token_dim == 3:  # char
        if token_name == "elmo":
            return _pad_char_tokens(
                tokens, max_len, char_maxlen=50, pad_value=261,
            )  # 260: padding_character, +1 for mask
        elif token_name == "char":
            return _pad_char_tokens(tokens, max_len, char_minlen=10, pad_value=pad_value)
        else:
            return _pad_char_tokens(tokens, max_len, char_minlen=1, pad_value=pad_value)
    else:
        return tokens


def get_token_dim(tokens, dim=0):
    if type(tokens) == torch.Tensor:
        dim = tokens.dim()
        if tokens.size(-1) > 1:
            dim += 1
        return dim

    if type(tokens) == np.ndarray:
        dim = tokens.ndim
        if tokens.shape[-1] > 1:
            dim += 1
        return dim

    if type(tokens) == list or type(tokens) == tuple:
        dim = get_token_dim(tokens[0], dim + 1)
    return dim


def get_token_type(tokens):
    token = tokens[0]
    while isinstance(token, np.ndarray) and isinstance(token, list):
        token = token[0]
    return type(token)


def is_lazy(tokens):
    if type(tokens) == list:
        tokens = tokens[0]

    if callable(tokens):
        return True
    else:
        return False


def transpose(list_of_dict, skip_keys=[]):
    if type(skip_keys) != list:
        raise ValueError(f"skip_keys type must be list. not {type(skip_keys)}")

    dict_of_list = defaultdict(lambda: [])
    for dic in list_of_dict:
        for key, value in dic.items():
            if key in skip_keys:
                continue
            dict_of_list[key].append(value)
    return dict_of_list


def sanity_check_iob(naive_tokens, tag_texts):
    """
    Check if the IOB tags are valid.

    * Args:
        naive_tokens: tokens split by .split()
        tag_texts: list of tags in IOB format
    """
    def prefix(tag):
        if tag == "O":
            return tag
        return tag.split("-")[0]

    def body(tag):
        if tag == "O":
            return None
        return tag.split("-")[1]

    # same number check
    assert len(naive_tokens) == len(tag_texts), \
        f"""Number of tokens and tags doest not match.
        original tokens: {naive_tokens}
        tags: {tag_texts}"""

    # IOB format check
    prev_tag = None
    for tag_text in tag_texts:
        curr_tag = tag_text

        if prev_tag is None:  # first tag
            assert prefix(curr_tag) in ["B", "O"], \
                f"""Wrong tag: first tag starts with I.
                tag: {curr_tag}"""""

        else:  # following tags
            if prefix(prev_tag) in ["B", "I"]:
                assert (
                        (prefix(curr_tag) == "I" and body(curr_tag) == body(prev_tag))
                        or (prefix(curr_tag) == "B")
                        or (prefix(curr_tag) == "O")
                ), f"""Wrong tag: following tag mismatch.
                    previous tag: {prev_tag}
                    current tag: {curr_tag}"""

            elif prefix(prev_tag) in ["O"]:
                assert prefix(curr_tag) in ["B", "O"], \
                    f"""Wrong tag: following tag mismatch.
                    previous tag: {prev_tag}
                    current tag: {curr_tag}"""
            else:
                raise RuntimeError(f"Encountered unknown tag: {prev_tag}.")

        prev_tag = curr_tag

def get_is_head_of_word(naive_tokens, sequence_tokens):
    """
    Return a list of flags whether the token is head(prefix) of naively split tokens
    
    ex) naive_tokens: ["hello.", "how", "are", "you?"]
        sequence_tokens: ["hello", ".", "how", "are", "you", "?"]
        
        => [1, 0, 1, 1, 1, 0]

    * Args:
        naive_tokens: a list of tokens, naively split by whitespace
        sequence_tokens: a list of tokens, split by 'word_tokenizer'

    * Returns:
        is_head_of_word: a list with its length the same as that of 'sequence_tokens'.
            has 1 if the tokenized word at the position is head(prefix) of a `naive_token`
            and 0 if otherwise.
    """
    
    is_head_of_word = []
    for naive_token in naive_tokens:
        consumed_chars = 0
        consumed_words = 0
        for sequence_token in sequence_tokens:
            if naive_token[consumed_chars:].startswith(sequence_token):
                is_head_of_word.append(0 if consumed_chars else 1)
                consumed_chars += len(sequence_token)
                consumed_words += 1
            else:
                break
        sequence_tokens = sequence_tokens[consumed_words:]
    return is_head_of_word
