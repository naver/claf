
import argparse
from argparse import RawTextHelpFormatter
import json
import os
import sys

import torch

from claf import nsml
from claf.config import utils
from claf.config.namespace import NestedNamespace
from claf.learn.mode import Mode


def config(argv=None, mode=None):
    if argv is None:
        argv = sys.argv[1:]  # 0 is excute file_name

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    general(parser)

    if mode == Mode.EVAL:
        evaluate(parser)
        return parser.parse_args(argv, namespace=NestedNamespace())

    if mode == Mode.PREDICT:
        predict(parser)
        return parser.parse_args(argv, namespace=NestedNamespace())

    if mode == Mode.MACHINE:
        machine(parser)
        config = parser.parse_args(argv, namespace=NestedNamespace())

        if config.machine_config is None:
            raise ValueError("--machine_config is required.")
        machine_config_path = os.path.join("machine_config", config.machine_config + ".json")
        with open(machine_config_path, "r") as f:
            defined_config = json.load(f)
        config.overwrite(defined_config)
        return config

    return train_config(parser, input_argv=argv)


def train_config(parser, input_argv=None):
    data(parser)
    token(parser)
    model(parser)
    if nsml.IS_ON_NSML:
        nsml_for_internal(parser)
    trainer(parser)

    # Use from config file
    base_config(parser)

    config = parser.parse_args(input_argv, namespace=NestedNamespace())

    use_base_config = config.base_config
    # use pre-defined base_config
    if use_base_config:
        base_config_path = os.path.join("base_config", config.base_config + ".json")
        with open(base_config_path, "r") as f:
            defined_config = json.load(f)
        # config.overwrite(defined_config)

        config = NestedNamespace()
        config.load_from_json(defined_config)

    # overwrite input argument when base_config and arguments are provided.
    # (eg. --base_config bidaf --learning_rate 2) -> set bidaf.json then overwrite learning_rate 2)
    input_args = get_input_arguments(parser, input_argv)
    for k, v in input_args.items():
        setattr(config, k, v)

    if not use_base_config:
        config = optimize_config(config)

    set_gpu_env(config)
    set_batch_size(config)
    return config


def get_input_arguments(parser, input_arguments):
    flat_config = parser.parse_args(input_arguments)
    config_dict = utils.convert_config2dict(flat_config)
    config_default_none = {k: None for k in config_dict.keys()}

    input_parser = argparse.ArgumentParser(parents=[parser], conflict_handler="resolve")
    input_parser.set_defaults(**config_default_none)

    input_config = input_parser.parse_args(input_arguments)
    input_config = utils.convert_config2dict(input_config)

    if "base_config" in input_config:
        del input_config["base_config"]
    return {k: v for k, v in input_config.items() if v is not None}


def optimize_config(config, is_test=False):
    if not is_test:
        # Remove unselected argument
        token_excepts = config.token.names + ["names", "types", "tokenizer"]
        config.delete_unselected(config.token, excepts=token_excepts)
        config.delete_unselected(config.model, excepts=["name", config.model.name])
        config.delete_unselected(
            config.optimizer,
            excepts=[
                "op_type",
                config.optimizer.op_type,
                "learning_rate",
                "lr_scheduler_type",
                config.optimizer.lr_scheduler_type,
                "exponential_moving_average",
            ],
        )

    return config


def set_gpu_env(config):
    # GPU & NSML
    config.use_gpu = torch.cuda.is_available() or nsml.IS_ON_NSML

    if nsml.IS_ON_NSML:
        if getattr(config, "nsml", None) is None:
            config.nsml = NestedNamespace()
        config.nsml.dataset_path = nsml.DATASET_PATH
        config.gpu_num = int(nsml.GPU_NUM)
    else:
        config.gpu_num = len(getattr(config, "cuda_devices", []))

    if not config.use_gpu:
        config.gpu_num = 0
        config.cuda_devices = None


def set_batch_size(config):
    # dynamic batch_size (multi-gpu and gradient_accumulation_steps)
    batch_size = config.iterator.batch_size
    if config.gpu_num > 1:
        batch_size *= config.gpu_num
    if getattr(config.optimizer, "gradient_accumulation_steps", None):
        batch_size = batch_size // config.optimizer.gradient_accumulation_steps
    config.iterator.batch_size = int(batch_size)


def arg_str2bool(v):
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# fmt: off
def general(parser):

    group = parser.add_argument_group("Genearl")
    group.add_argument(
        "--seed_num",
        type=int, default=21, dest="seed_num",
        help=""" Manually set seed_num (Python, Numpy, Pytorch) default is 21 """,
    )
    group.add_argument(
        "--cuda_devices", nargs="+",
        type=int, default=[], dest="cuda_devices",
        help=""" Set cuda_devices ids (use GPU). if you use NSML, use GPU_NUM""",
    )
    group.add_argument(
        "--slack_url",
        type=str, default=None, dest="slack_url",
        help=""" Slack notification (Incoming Webhook) """,
    )


def data(parser):

    group = parser.add_argument_group("Data Reader")
    group.add_argument(
        "--dataset",
        type=str, default="squad", dest="data_reader.dataset",
        help=""" Dataset Name [squad|squad2] """,
    )
    group.add_argument(
        "--train_file_path",
        type=str, default="train-v1.1.json", dest="data_reader.train_file_path",
        help=""" train file path. """,
    )
    group.add_argument(
        "--valid_file_path",
        type=str, default="dev-v1.1.json", dest="data_reader.valid_file_path",
        help=""" validation file path. """,
    )
    group.add_argument(
        "--test_file_path",
        type=str, default=None, dest="data_reader.test_file_path",
        help=""" test file path. """,
    )

    group = parser.add_argument_group("  # SQuAD DataSet")
    group.add_argument(
        "--squad.context_max_length",
        type=int, default=None, dest="data_reader.squad.context_max_length",
        help=""" The number of SQuAD Context maximum length. """,
    )

    group = parser.add_argument_group("  # HistoryQA DataSet")
    group.add_argument(
        "--history.context_max_length",
        type=int, default=None, dest="data_reader.history.context_max_length",
        help=""" The number of HistoryQA Context maximum length. """,
    )

    group = parser.add_argument_group("  # SeqCls DataSet")
    group.add_argument(
        "--seq_cls.class_key",
        type=int, default=None, dest="data_reader.seq_cls.class_key",
        help=""" Name of the label to use for classification. """,
    )
    group.add_argument(
        "--seq_cls.sequence_max_length",
        type=int, default=None, dest="data_reader.seq_cls.sequence_max_length",
        help=""" The number of maximum sequence length. """,
    )

    group = parser.add_argument_group("  # SeqClsBert DataSet")
    group.add_argument(
        "--seq_cls_bert.class_key",
        type=int, default=None, dest="data_reader.seq_cls_bert.class_key",
        help=""" Name of the label to use for classification. """,
    )
    group.add_argument(
        "--seq_cls_bert.sequence_max_length",
        type=int, default=None, dest="data_reader.seq_cls_bert.sequence_max_length",
        help=""" The number of maximum sequence length. """,
    )

    group = parser.add_argument_group("  # TokClsBert DataSet")
    group.add_argument(
        "--tok_cls_bert.tag_key",
        type=int, default=None, dest="data_reader.tok_cls_bert.tag_key",
        help=""" Name of the label to use for classification. """,
    )
    group.add_argument(
        "--tok_cls_bert.ignore_tag_idx",
        type=int, default=None, dest="data_reader.tok_cls_bert.ignore_tag_idx",
        help=""" Index of the tag to ignore when calculating loss. (tag pad value) """,
    )
    group.add_argument(
        "--tok_cls_bert.sequence_max_length",
        type=int, default=None, dest="data_reader.tok_cls_bert.sequence_max_length",
        help=""" The number of maximum sequence length. """,
    )

    group = parser.add_argument_group("Iterator")
    group.add_argument(
        "--batch_size", type=int, default=32, dest="iterator.batch_size",
        help=""" Maximum batch size for trainer""",
    )


def token(parser):

    group = parser.add_argument_group("Token")
    group.add_argument(
        "--token_names", nargs="+",
        type=str, default=["char", "word"], dest="token.names",
        help=""" Define tokens name""",
    )
    group.add_argument(
        "--token_types", nargs="+",
        type=str, default=["char", "word"], dest="token.types",
        help="""\
    Use pre-defined token
    (tokenizer -> indexer -> embedder)

    [char|cove|elmo|exact_match|frequent_word|word]""",
    )

    group = parser.add_argument_group(" # Vocabulary")

    group.add_argument(
        "--char.pad_token",
        type=str, default=None, dest="token.char.vocab.pad_token",
        help=""" Padding Token value""",
    )
    group.add_argument(
        "--char.oov_token",
        type=str, default=None, dest="token.char.vocab.oov_token",
        help=""" Out-of-Vocabulary Token value""",
    )
    group.add_argument(
        "--char.start_token",
        type=str, default=None, dest="token.char.vocab.start_token",
        help=""" Start Token value""",
    )
    group.add_argument(
        "--char.end_token",
        type=str, default=None, dest="token.char.vocab.end_token",
        help=""" End Token value""",
    )
    group.add_argument(
        "--char.min_count",
        type=int, default=None, dest="token.char.vocab.min_count",
        help=""" The number of token's min count""",
    )
    group.add_argument(
        "--char.max_vocab_size",
        type=int, default=260, dest="token.char.vocab.max_vocab_size",
        help=""" The number of vocab's max size""",
    )

    group.add_argument(
        "--word.pad_token",
        type=str, default=None, dest="token.word.vocab.pad_token",
        help=""" Padding Token value""",
    )
    group.add_argument(
        "--word.oov_token",
        type=str, default=None, dest="token.word.vocab.oov_token",
        help=""" Out-of-Vocabulary Token value""",
    )
    group.add_argument(
        "--word.min_count",
        type=int, default=None, dest="token.word.vocab.min_count",
        help=""" The number of token's min count""",
    )
    group.add_argument(
        "--word.max_vocab_size",
        type=int, default=None, dest="token.word.vocab.max_vocab_size",
        help=""" The number of vocab's max size""",
    )

    group.add_argument(
        "--frequent_word.frequent_count",
        type=int, default=1000, dest="token.frequent_word.vocab.frequent_count",
        help="""\
    The number of threshold frequent count
    (>= threshold -> fine-tune, < threshold -> fixed)""",
    )

    group = parser.add_argument_group(" # Tokenizer")
    group.add_argument(
        "--tokenizer.char.name",
        type=str, default="character", dest="token.tokenizer.char.name",
        help="""\
    CharTokenizer package name [character|jamo_ko]
    Default is 'character' """,
    )
    group.add_argument(
        "--tokenizer.subword.name",
        type=str, default="wordpiece", dest="token.tokenizer.subword.name",
        help="""\
    SubWordTokenizer package name [wordpiece]
    Default is 'wordpiece' """,
    )
    group.add_argument(
        "--tokenizer.subword.wordpiece.do_lower_case",
        type=arg_str2bool, default=True, dest="token.tokenizer.subword.wordpiece.do_lower_case",
        help="""\
    Wordpiece Tokenizer do_lower_case or not
    Default is 'True' """,
    )
    group.add_argument(
        "--tokenizer.word.name",
        type=str, default="treebank_en", dest="token.tokenizer.word.name",
        help="""\
    WordTokenizer package name [treebank_en|spacy_en|mecab_ko]
    Default is 'treebank_en' """,
    )
    group.add_argument(
        "--tokenizer.word.split_with_regex",
        type=arg_str2bool, default=False, dest="token.tokenizer.word.split_with_regex",
        help=""" preprocess for SQuAD Context data (simple regex) """,
    )
    group.add_argument(
        "--tokenizer.sent.name",
        type=str, default="punkt", dest="token.tokenizer.sent.name",
        help="""\
    SentTokenizer package name [punkt]
    Default is 'punkt' """,
    )

    group = parser.add_argument_group(" # Indexer")
    group.add_argument(
        "--char.insert_char_start",
        type=arg_str2bool, default=False, dest="token.char.indexer.insert_char_start",
        help=""" insert first start_token to tokens""",
    )
    group.add_argument(
        "--char.insert_char_end",
        type=arg_str2bool, default=False, dest="token.char.indexer.insert_char_end",
        help=""" append end_token to tokens""",
    )

    group.add_argument(
        "--exact_match.lower",
        type=arg_str2bool, default=True, dest="token.exact_match.indexer.lower",
        help=""" add lower case feature """,
    )
    group.add_argument(
        "--exact_match.lemma",
        type=arg_str2bool, default=True, dest="token.exact_match.indexer.lemma",
        help=""" add lemma case feature """,
    )

    group.add_argument(
        "--linguistic.pos_tag",
        type=arg_str2bool, default=True, dest="token.linguistic.indexer.pos_tag",
        help=""" add POS Tagging feature """,
    )
    group.add_argument(
        "--linguistic.ner",
        type=arg_str2bool, default=True, dest="token.linguistic.indexer.ner",
        help=""" add Named Entity Recognition feature """,
    )
    group.add_argument(
        "--linguistic.dep",
        type=arg_str2bool, default=False, dest="token.linguistic.indexer.dep",
        help=""" add Dependency Parser feature """,
    )

    group.add_argument(
        "--word.lowercase",
        type=arg_str2bool, default=False, dest="token.word.indexer.lowercase",
        help=""" Apply word token to lowercase""",
    )
    group.add_argument(
        "--word.insert_start",
        type=arg_str2bool, default=False, dest="token.word.indexer.insert_start",
        help=""" insert first start_token to tokens""",
    )
    group.add_argument(
        "--word.insert_end",
        type=arg_str2bool, default=False, dest="token.word.indexer.insert_end",
        help=""" append end_token to tokens""",
    )

    group = parser.add_argument_group(" # Embedding")

    group.add_argument(
        "--char.embed_dim",
        type=int, default=16, dest="token.char.embedding.embed_dim",
        help=""" The number of Embedding dimension""",
    )
    group.add_argument(
        "--char.kernel_sizes", nargs="+",
        type=int, default=[5], dest="token.char.embedding.kernel_sizes",
        help=""" CharCNN kernel_sizes (n-gram)""",
    )
    group.add_argument(
        "--char.num_filter",
        type=int, default=100, dest="token.char.embedding.num_filter",
        help=""" The number of CNN filter""",
    )
    group.add_argument(
        "--char.activation",
        type=str, default="relu", dest="token.char.embedding.activation",
        help=""" CharCNN activation Function (default: ReLU)""",
    )
    group.add_argument(
        "--char.dropout",
        type=float, default=0.2, dest="token.char.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.2)""",
    )

    group.add_argument(
        "--cove.glove_pretrained_path",
        type=str, default=None, dest="token.cove.embedding.glove_pretrained_path",
        help=""" CoVe's word embedding pretrained_path (GloVE 840B.300d)""",
    )
    group.add_argument(
        "--cove.model_pretrained_path",
        type=str, default=None, dest="token.cove.embedding.model_pretrained_path",
        help=""" CoVe Model pretrained_path """,
    )
    group.add_argument(
        "--cove.trainable",
        type=arg_str2bool, default=True, dest="token.cove.embedding.trainable",
        help=""" CoVe Embedding Trainable""",
    )
    group.add_argument(
        "--cove.dropout",
        type=float, default=0.2, dest="token.cove.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.2)""",
    )
    group.add_argument(
        "--cove.project_dim",
        type=int, default=None, dest="token.cove.embedding.project_dim",
        help=""" The number of projection dimension""",
    )

    group.add_argument(
        "--elmo.options_file",
        type=str, default="elmo_2x4096_512_2048cnn_2xhighway_options.json", dest="token.elmo.embedding.options_file",
        help=""" The option file path of ELMo""",
    )
    group.add_argument(
        "--elmo.weight_file",
        type=str, default="elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", dest="token.elmo.embedding.weight_file",
        help=""" The weight file path of ELMo""",
    )
    group.add_argument(
        "--elmo.trainable",
        type=arg_str2bool, default=False, dest="token.elmo.embedding.trainable",
        help=""" elmo Embedding Trainable""",
    )
    group.add_argument(
        "--elmo.dropout",
        type=float, default=0.5, dest="token.elmo.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.5)""",
    )
    group.add_argument(
        "--elmo.project_dim",
        type=int, default=None, dest="token.elmo.embedding.project_dim",
        help=""" The number of projection dimension (default is None)""",
    )

    group.add_argument(
        "--word_permeability.memory_clip",
        type=int, default=3, dest="token.word_permeability.embedding.memory_clip",
        help=""" The number of memory cell clip value """,
    )
    group.add_argument(
        "--word_permeability.proj_clip",
        type=int, default=3, dest="token.word_permeability.embedding.proj_clip",
        help=""" The number of p clip value after projection """,
    )
    group.add_argument(
        "--word_permeability.embed_dim",
        type=int, default=1024, dest="token.word_permeability.embedding.embed_dim",
        help=""" The number of Embedding dimension""",
    )
    group.add_argument(
        "--word_permeability.linear_dim",
        type=int, default=None, dest="token.word_permeability.embedding.linear_dim",
        help=""" The number of linear projection dimension""",
    )
    group.add_argument(
        "--word_permeability.trainable",
        type=arg_str2bool, default=False, dest="token.word_permeability.embedding.trainable",
        help=""" word_permeability Embedding Trainable """,
    )
    group.add_argument(
        "--word_permeability.dropout",
        type=float, default=0.5, dest="token.word_permeability.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.5)""",
    )
    group.add_argument(
        "--word_permeability.activation",
        type=str, default="tanh", dest="token.word_permeability.embedding.activation",
        help=""" Activation Function (default is 'tanh') """,
    )
    group.add_argument(
        "--word_permeability.bidirectional",
        type=arg_str2bool, default=False, dest="token.word_permeability.embedding.bidirectional",
        help=""" bidirectional use or not ([forward;backward]) (default is False) """,
    )

    group.add_argument(
        "--frequent_word.embed_dim",
        type=int, default=100, dest="token.frequent_word.embedding.embed_dim",
        help=""" The number of Embedding dimension""",
    )
    group.add_argument(
        "--frequent_word.pretrained_path",
        type=str, default=None, dest="token.frequent_word.embedding.pretrained_path",
        help=""" Add pretrained Word vector model's path. (support file format like Glove)""",
    )
    group.add_argument(
        "--frequent_word.dropout",
        type=float, default=0.2, dest="token.frequent_word.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.2)""",
    )

    group.add_argument(
        "--word.embed_dim",
        type=int, default=100, dest="token.word.embedding.embed_dim",
        help=""" The number of Embedding dimension""",
    )
    group.add_argument(
        "--word.pretrained_path",
        type=str, default=None, dest="token.word.embedding.pretrained_path",
        help=""" Add pretrained word vector model's path. (support file format like Glove)""",
    )
    group.add_argument(
        "--word.trainable",
        type=arg_str2bool, default=True, dest="token.word.embedding.trainable",
        help=""" Word Embedding Trainable""",
    )
    group.add_argument(
        "--word.dropout",
        type=float, default=0.2, dest="token.word.embedding.dropout",
        help=""" Embedding dropout prob (default: 0.2)""",
    )


def model(parser):

    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model_name",
        type=str, default="bidaf", dest="model.name",
        help="""\

    Pre-defined model

    * Reading Comprehension
      [bert_for_qa|bidaf|bidaf_no_answer|docqa|docqa_no_answer|dclaf|qanet|simple]

    * Semantic Parsing
      [sqlnet]

    * Sequence Classification
      [bert_for_seq_cls|structured_self_attention]

    * Token Classification
      [bert_for_tok_cls]
    """,
    )

    reading_comprehension_title = "ㅁReading Comprehension"
    group = parser.add_argument_group(f"{reading_comprehension_title}\n # BERT for QuestionAnswering")
    group.add_argument(
        "--bert_for_qa.pretrained_model_name",
        type=str, default=None, dest="model.bert_for_qa.pretrained_model_name",
        help=""" A str with the name of a pre-trained model to load selected in the list of (default: None):
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese` """,
    )
    group.add_argument(
        "--bert_for_qa.answer_maxlen",
        type=int, default=None, dest="model.bert_for_qa.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )

    group = parser.add_argument_group(f" # BiDAF")
    group.add_argument(
        "--bidaf.aligned_query_embedding",
        type=int, default=False, dest="model.bidaf.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: False)""",
    )
    group.add_argument(
        "--bidaf.answer_maxlen",
        type=int, default=None, dest="model.bidaf.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )
    group.add_argument(
        "--bidaf.model_dim",
        type=int, default=100, dest="model.bidaf.model_dim",
        help=""" The number of BiDAF model dimension""",
    )
    group.add_argument(
        "--bidaf.contextual_rnn_num_layer",
        type=int, default=1, dest="model.bidaf.contextual_rnn_num_layer",
        help=""" The number of BiDAF model contextual_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf.modeling_rnn_num_layer",
        type=int, default=2, dest="model.bidaf.modeling_rnn_num_layer",
        help=""" The number of BiDAF model modeling_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf.predict_rnn_num_layer",
        type=int, default=1, dest="model.bidaf.predict_rnn_num_layer",
        help=""" The number of BiDAF model predict_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf.dropout",
        type=float, default=0.2, dest="model.bidaf.dropout",
        help=""" The prob of BiDAF dropout""",
    )

    group = parser.add_argument_group(" # BiDAF + Simple bias")
    group.add_argument(
        "--bidaf_no_answer.aligned_query_embedding",
        type=int, default=False, dest="model.bidaf_no_answer.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: False)""",
    )
    group.add_argument(
        "--bidaf_no_answer.answer_maxlen",
        type=int, default=None, dest="model.bidaf_no_answer.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )
    group.add_argument(
        "--bidaf_no_answer.model_dim",
        type=int, default=100, dest="model.bidaf_no_answer.model_dim",
        help=""" The number of BiDAF model dimension""",
    )
    group.add_argument(
        "--bidaf_no_answer.contextual_rnn_num_layer",
        type=int, default=1, dest="model.bidaf_no_answer.contextual_rnn_num_layer",
        help=""" The number of BiDAF model contextual_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf_no_answer.modeling_rnn_num_layer",
        type=int, default=2, dest="model.bidaf_no_answer.modeling_rnn_num_layer",
        help=""" The number of BiDAF model modeling_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf_no_answer.predict_rnn_num_layer",
        type=int, default=1, dest="model.bidaf_no_answer.predict_rnn_num_layer",
        help=""" The number of BiDAF model predict_rnn's recurrent layers""",
    )
    group.add_argument(
        "--bidaf_no_answer.dropout",
        type=float, default=0.2, dest="model.bidaf_no_answer.dropout",
        help=""" The prob of BiDAF dropout""",
    )

    group = parser.add_argument_group(" # Simple")
    group.add_argument(
        "--simple.answer_maxlen",
        type=int, default=None, dest="model.simple.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )
    group.add_argument(
        "--simple.model_dim",
        type=int, default=100, dest="model.simple.model_dim",
        help=""" The number of Simple model dimension""",
    )
    group.add_argument(
        "--simple.dropout",
        type=float, default=0.2, dest="model.simple.dropout",
        help=""" The prob of Simple dropout""",
    )

    group = parser.add_argument_group(" # QANet")
    group.add_argument(
        "--qanet.aligned_query_embedding",
        type=int, default=False, dest="model.qanet.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: False)""",
    )
    group.add_argument(
        "--qanet.answer_maxlen",
        type=int, default=30, dest="model.qanet.answer_maxlen",
        help=""" The number of maximum answer's length (default: 30)""",
    )
    group.add_argument(
        "--qanet.model_dim",
        type=int, default=128, dest="model.qanet.model_dim",
        help=""" The number of QANet model dimension""",
    )
    group.add_argument(
        "--qanet.kernel_size_in_embedding",
        type=int, default=7, dest="model.qanet.kernel_size_in_embedding",
        help=""" The number of QANet model Embed Encoder kernel_size""",
    )
    group.add_argument(
        "--qanet.num_head_in_embedding",
        type=int, default=8, dest="model.qanet.num_head_in_embedding",
        help=""" The number of QANet model Multi-Head Attention's head in Embedding Block""",
    )
    group.add_argument(
        "--qanet.num_conv_block_in_embedding",
        type=int, default=4, dest="model.qanet.num_conv_block_in_embedding",
        help=""" The number of QANet model Conv Blocks in Embedding Block""",
    )
    group.add_argument(
        "--qanet.num_embedding_encoder_block",
        type=int, default=1, dest="model.qanet.num_embedding_encoder_block",
        help=""" The number of QANet model Embedding Encoder Blocks""",
    )
    group.add_argument(
        "--qanet.kernel_size_in_modeling",
        type=int, default=5, dest="model.qanet.kernel_size_in_modeling",
        help=""" The number of QANet model Model Encoder kernel_size""",
    )
    group.add_argument(
        "--qanet.num_head_in_modeling",
        type=int, default=8, dest="model.qanet.num_head_in_modeling",
        help=""" The number of QANet model Multi-Head Attention's head in Modeling Block""",
    )
    group.add_argument(
        "--qanet.num_conv_block_in_modeling",
        type=int, default=2, dest="model.qanet.num_conv_block_in_modeling",
        help=""" The number of QANet model Conv Blocks in Modeling Block""",
    )
    group.add_argument(
        "--qanet.num_modeling_encoder_block",
        type=int, default=7, dest="model.qanet.num_modeling_encoder_block",
        help=""" The number of QANet model Modeling Encoder Blocks""",
    )
    group.add_argument(
        "--qanet.layer_dropout",
        type=float, default=0.9, dest="model.qanet.layer_dropout",
        help=""" The prob of QANet model layer dropout""",
    )
    group.add_argument(
        "--qanet.dropout",
        type=float, default=0.1, dest="model.qanet.dropout",
        help=""" The prob of QANet dropout""",
    )

    group = parser.add_argument_group(" # DocQA")
    group.add_argument(
        "--docqa.aligned_query_embedding",
        type=arg_str2bool, default=False, dest="model.docqa.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: False)""",
    )
    group.add_argument(
        "--docqa.answer_maxlen",
        type=int, default=17, dest="model.docqa.answer_maxlen",
        help=""" The number of maximum answer's length (default: 17)""",
    )
    group.add_argument(
        "--docqa.rnn_dim",
        type=int, default=100, dest="model.docqa.rnn_dim",
        help=""" The number of DocQA model rnn dimension""",
    )
    group.add_argument(
        "--docqa.linear_dim",
        type=int, default=200, dest="model.docqa.linear_dim",
        help=""" The number of DocQA model linear dimension""",
    )
    group.add_argument(
        "--docqa.preprocess_rnn_num_layer",
        type=int, default=1, dest="model.docqa.preprocess_rnn_num_layer",
        help=""" The number of DocQA model preprocess_rnn's recurrent layers""",
    )
    group.add_argument(
        "--docqa.modeling_rnn_num_layer",
        type=int, default=1, dest="model.docqa.modeling_rnn_num_layer",
        help=""" The number of DocQA model modeling_rnn's recurrent layers""",
    )
    group.add_argument(
        "--docqa.predict_rnn_num_layer",
        type=int, default=1, dest="model.docqa.predict_rnn_num_layer",
        help=""" The number of DocQA model predict_rnn's recurrent layers""",
    )
    group.add_argument(
        "--docqa.dropout",
        type=float, default=0.2, dest="model.docqa.dropout",
        help=""" The prob of DocQA dropout""",
    )
    group.add_argument(
        "--docqa.weight_init",
        type=arg_str2bool, default=True, dest="model.docqa.weight_init",
        help=""" Weight Init""",
    )

    group = parser.add_argument_group(" # DocQA + No_Answer Option")
    group.add_argument(
        "--docqa_no_answer.aligned_query_embedding",
        type=arg_str2bool, default=False, dest="model.docqa_no_answer.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: False)""",
    )
    group.add_argument(
        "--docqa_no_answer.answer_maxlen",
        type=int, default=17, dest="model.docqa_no_answer.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )
    group.add_argument(
        "--docqa_no_answer.rnn_dim",
        type=int, default=100, dest="model.docqa_no_answer.rnn_dim",
        help=""" The number of docqa_no_answer model rnn dimension""",
    )
    group.add_argument(
        "--docqa_no_answer.linear_dim",
        type=int, default=200, dest="model.docqa_no_answer.linear_dim",
        help=""" The number of docqa_no_answer model linear dimension""",
    )
    group.add_argument(
        "--docqa_no_answer.dropout",
        type=float, default=0.2, dest="model.docqa_no_answer.dropout",
        help=""" The prob of QANet dropout""",
    )
    group.add_argument(
        "--docqa_no_answer.weight_init",
        type=arg_str2bool, default=True, dest="model.docqa_no_answer.weight_init",
        help=""" Weight Init""",
    )

    group = parser.add_argument_group(" # Dclaf")
    group.add_argument(
        "--dclaf.aligned_query_embedding",
        type=int, default=True, dest="model.dclaf.aligned_query_embedding",
        help=""" Aligned Question Embedding  (default: True)""",
    )
    group.add_argument(
        "--dclaf.answer_maxlen",
        type=int, default=15, dest="model.dclaf.answer_maxlen",
        help=""" The number of maximum answer's length (default: None)""",
    )
    group.add_argument(
        "--dclaf.model_dim",
        type=int, default=128, dest="model.dclaf.model_dim",
        help=""" The number of document reader model dimension""",
    )
    group.add_argument(
        "--dclaf.dropout",
        type=int, default=0.3, dest="model.dclaf.dropout",
        help=""" The number of document reader model dropout""",
    )

    semantic_parsing_title = "ㅁSemantic Parsing"
    group = parser.add_argument_group(f"{semantic_parsing_title}\n # SQLNet")
    group.add_argument(
        "--sqlnet.column_attention",
        type=int, default=True, dest="model.sqlnet.column_attention",
        help=""" Compute attention map on a question conditioned on the column names (default: True)""",
    )
    group.add_argument(
        "--sqlnet.model_dim",
        type=int, default=100, dest="model.sqlnet.model_dim",
        help=""" The number of document reader model dimension""",
    )
    group.add_argument(
        "--sqlnet.rnn_num_layer",
        type=int, default=2, dest="model.sqlnet.rnn_num_layer",
        help=""" The number of SQLNet model rnn's recurrent layers""",
    )
    group.add_argument(
        "--sqlnet.dropout",
        type=int, default=0.3, dest="model.sqlnet.dropout",
        help=""" The prob of model dropout """,
    )
    group.add_argument(
        "--sqlnet.column_maxlen",
        type=int, default=4, dest="model.sqlnet.column_maxlen",
        help=""" The number of maximum column's length (default: 4)""",
    )
    group.add_argument(
        "--sqlnet.token_maxlen",
        type=int, default=200, dest="model.sqlnet.token_maxlen",
        help=""" An upper-bound N on the number of decoder tokeni """,
    )
    group.add_argument(
        "--sqlnet.conds_column_loss_alpha",
        type=int, default=0.3, dest="model.sqlnet.conds_column_loss_alpha",
        help=""" balance the positive data versus negative data """,
    )

    sequence_classification_title = "ㅁSequence Classification"
    group = parser.add_argument_group(f"{sequence_classification_title}\n # BERT for Sequence Classification")
    group.add_argument(
        "--bert_for_seq_cls.pretrained_model_name",
        type=str, default=None, dest="model.bert_for_seq_cls.pretrained_model_name",
        help=""" A str with the name of a pre-trained model to load selected in the list of (default: None):
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese` """,
    )
    group.add_argument(
        "--bert_for_seq_cls.dropout",
        type=float, default=0.2, dest="model.bert_for_seq_cls.dropout",
        help=""" The prob of fc layer dropout """
    )

    group = parser.add_argument_group(f"{sequence_classification_title}\n # Structured Self Attention")
    group.add_argument(
        "--structured_self_attention.token_encoder",
        type=str, default="bilstm", dest="model.structured_self_attention.token_encoder",
        help=""" Token encoder type [none|bilstm] """
    )
    group.add_argument(
        "--structured_self_attention.encoding_rnn_hidden_dim",
        type=int, default=600, dest="model.structured_self_attention.encoding_rnn_hidden_dim",
        help=""" The number of hidden dimension for each token """
    )
    group.add_argument(
        "--structured_self_attention.encoding_rnn_num_layer",
        type=int, default=2, dest="model.structured_self_attention.encoding_rnn_num_layer",
        help=""" The number of layers of token encoding rnn """
    )
    group.add_argument(
        "--structured_self_attention.encoding_rnn_dropout",
        type=float, default=0., dest="model.structured_self_attention.encoding_rnn_dropout",
        help=""" The prob of token encoding rnn dropout (between layers) """
    )
    group.add_argument(
        "--structured_self_attention.attention_dim",
        type=int, default=350, dest="model.structured_self_attention.attention_dim",
        help=""" The number of embedding dimension for attention """
    )
    group.add_argument(
        "--structured_self_attention.num_attention_heads",
        type=int, default=30, dest="model.structured_self_attention.num_attention_heads",
        help=""" The number of rows for attention (attention heads) """
    )
    group.add_argument(
        "--structured_self_attention.project_dim",
        type=int, default=2000, dest="model.structured_self_attention.project_dim",
        help=""" The number of bottleneck layer embedding dimension """
    )
    group.add_argument(
        "--structured_self_attention.dropout",
        type=float, default=0.5, dest="model.structured_self_attention.dropout",
        help=""" The prob of bottleneck-making fnn dropout """
    )
    group.add_argument(
        "--structured_self_attention.penalization_coefficient",
        type=float, default=1., dest="model.structured_self_attention.penalization_coefficient",
        help=""" The coefficient of penalization term """
    )

    token_classification_title = "ㅁToken Classification"
    group = parser.add_argument_group(f"{token_classification_title}\n # BERT for Token Classification")
    group.add_argument(
        "--bert_for_tok_cls.pretrained_model_name",
        type=str, default=None, dest="model.bert_for_tok_cls.pretrained_model_name",
        help=""" A str with the name of a pre-trained model to load selected in the list of (default: None):
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese` """,
    )
    group.add_argument(
        "--bert_for_tok_cls.dropout",
        type=float, default=0.2, dest="model.bert_for_tok_cls.dropout",
        help=""" The prob of fc layer dropout """
    )


def nsml_for_internal(parser):

    group = parser.add_argument_group("NSML")
    group.add_argument(
        "--pause",
        type=int, default=0, dest="nsml.pause",
        help=""" NSML default setting"""
    )
    group.add_argument(
        "--iteration",
        type=int, default=0, dest="nsml.iteration",
        help=""" Start from NSML epoch count""",
    )


def trainer(parser):

    group = parser.add_argument_group("Trainer")
    group.add_argument(
        "--num_epochs",
        type=int, default=20, dest="trainer.num_epochs",
        help=""" The number of training epochs""",
    )
    group.add_argument(
        "--patience",
        type=int, default=10, dest="trainer.early_stopping_threshold",
        help=""" The number of early stopping threshold""",
    )
    group.add_argument(
        "--metric_key",
        type=str, default="em", dest="trainer.metric_key",
        help=""" The key of metric for model's score""",
    )
    group.add_argument(
        "--verbose_step_count",
        type=int, default=100, dest="trainer.verbose_step_count",
        help=""" The number of training verbose""",
    )
    group.add_argument(
        "--save_epoch_count",
        type=int, default=1, dest="trainer.save_epoch_count",
        help=""" The number of save epoch count""",
    )
    group.add_argument(
        "--log_dir",
        type=str, default="logs/experiment_1", dest="trainer.log_dir",
        help=""" TensorBoard and Checkpoint log directory""",
    )

    group = parser.add_argument_group("Gradient")
    group.add_argument(
        "--grad_max_norm",
        type=float, default=None, dest="trainer.grad_max_norm",
        help=""" Clips gradient norm of an iterable of parameters. ()Default: None)""")

    group = parser.add_argument_group("Optimizer")
    group.add_argument(
        "--optimizer_type",
        type=str, default="adam", dest="optimizer.op_type",
        help=""" Optimizer
    (https://pytorch.org/docs/stable/optim.html#algorithms)

    - adadelta: ADADELTA: An Adaptive Learning Rate Method
        (https://arxiv.org/abs/1212.5701)
    - adagrad: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
        (http://jmlr.org/papers/v12/duchi11a.html)
    - adam: Adam: A Method for Stochastic Optimization
        (https://arxiv.org/abs/1412.6980)
    - sparse_adam: Implements lazy version of Adam algorithm suitable for sparse tensors.
        In this variant, only moments that show up in the gradient get updated,
        and only those portions of the gradient get applied to the parameters.
    - adamax: Implements Adamax algorithm (a variant of Adam based on infinity norm).
    - averaged_sgd: Acceleration of stochastic approximation by averaging
        (http://dl.acm.org/citation.cfm?id=131098)
    - rmsprop: Implements RMSprop algorithm.
        (https://arxiv.org/pdf/1308.0850v5.pdf)
    - rprop: Implements the resilient backpropagation algorithm.
    - sgd: Implements stochastic gradient descent (optionally with momentum).
        Nesterov momentum: (http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)

    [adadelta|adagrad|adam|sparse_adam|adamax|averaged_sgd|rmsprop|rprop|sgd]""",
    )
    group.add_argument(
        "--learning_rate",
        type=float, default=0.5, dest="optimizer.learning_rate",
        help="""\
    Starting learning rate.
    Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001 """,
    )

    group = parser.add_argument_group("  # Adadelta")
    group.add_argument(
        "--adadelta.rho",
        type=float, default=0.9, dest="optimizer.adadelta.rho",
        help="""\
    coefficient used for computing a running average of squared gradients
    Default: 0.9 """,
    )
    group.add_argument(
        "--adadelta.eps",
        type=float, default=1e-6, dest="optimizer.adadelta.eps",
        help="""\
    term added to the denominator to improve numerical stability
    Default: 1e-6 """,
    )
    group.add_argument(
        "--adadelta.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adadelta.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # Adagrad")
    group.add_argument(
        "--adagrad.lr_decay",
        type=float, default=0, dest="optimizer.adagrad.lr_decay",
        help="""\
    learning rate decay
    Default: 0 """,
    )
    group.add_argument(
        "--adagrad.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adagrad.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # Adam")
    group.add_argument(
        "--adam.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.adam.betas",
        help="""\
    coefficients used for computing running averages of gradient and its square
    Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--adam.eps",
        type=float, default=1e-8, dest="optimizer.adam.eps",
        help="""\
    term added to the denominator to improve numerical stability
    Default: 1e-8 """,
    )
    group.add_argument(
        "--adam.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adam.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # SparseAdam")
    group.add_argument(
        "--sparse_adam.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.sparse_adam.betas",
        help="""\
    coefficients used for computing running averages of gradient and its square
    Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--sparse_adam.eps",
        type=float, default=1e-8, dest="optimizer.sparse_adam.eps",
        help="""\
    term added to the denominator to improve numerical stability
    Default: 1e-8 """,
    )

    group = parser.add_argument_group("  # Adamax")
    group.add_argument(
        "--adamax.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.adamax.betas",
        help="""\
    coefficients used for computing running averages of gradient and its square.
    Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--adamax.eps",
        type=float, default=1e-8, dest="optimizer.adamax.eps",
        help="""\
    term added to the denominator to improve numerical stability.
    Default: 1e-8 """,
    )
    group.add_argument(
        "--adamax.weight_decay",
        type=float, default=0, dest="optimizer.adamax.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # ASGD (Averaged Stochastic Gradient Descent)")
    group.add_argument(
        "--averaged_sgd.lambd",
        type=float, default=1e-4, dest="optimizer.averaged_sgd.lambd",
        help="""\
    decay term
    Default: 1e-4 """,
    )
    group.add_argument(
        "--averaged_sgd.alpha",
        type=float, default=0.75, dest="optimizer.averaged_sgd.alpha",
        help="""\
    power for eta update
    Default: 0.75 """,
    )
    group.add_argument(
        "--averaged_sgd.t0",
        type=float, default=1e6, dest="optimizer.averaged_sgd.t0",
        help="""\
    point at which to start averaging
    Default: 1e6 """,
    )
    group.add_argument(
        "--averaged_sgd.weight_decay",
        type=float, default=0, dest="optimizer.averaged_sgd.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # RMSprop")
    group.add_argument(
        "--rmsprop.momentum",
        type=float, default=0, dest="optimizer.rmsprop.momentum",
        help="""\
    momentum factor
    Default: 0 """,
    )
    group.add_argument(
        "--rmsprop.alpha",
        type=float, default=0.99, dest="optimizer.rmsprop.alpha",
        help="""\
    smoothing constant
    Default: 0.99 """,
    )
    group.add_argument(
        "--rmsprop.eps",
        type=float, default=1e-8, dest="optimizer.rmsprop.eps",
        help="""\
    term added to the denominator to improve numerical stability.
    Default: 1e-8 """,
    )
    group.add_argument(
        "--rmsprop.centered",
        type=arg_str2bool, default=False, dest="optimizer.rmsprop.centered",
        help="""\
    if True, compute the centered RMSProp,
    the gradient is normalized by an estimation of its variance
    Default: False """,
    )
    group.add_argument(
        "--rmsprop.weight_decay",
        type=float, default=0, dest="optimizer.rmsprop.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("  # SGD (Stochastic Gradient Descent)")
    group.add_argument(
        "--sgd.momentum",
        type=float, default=0, dest="optimizer.sgd.momentum",
        help="""\
    momentum factor
    Default: 0 """,
    )
    group.add_argument(
        "--sgd.dampening",
        type=float, default=0, dest="optimizer.sgd.dampening",
        help="""\
    dampening for momentum
    Default: 0 """,
    )
    group.add_argument(
        "--sgd.nesterov",
        type=arg_str2bool, default=False, dest="optimizer.sgd.nesterov",
        help="""\
    enables Nesterov momentum
    Default: False """,
    )
    group.add_argument(
        "--sgd.weight_decay",
        type=float, default=0, dest="optimizer.sgd.weight_decay",
        help="""\
    weight decay (L2 penalty)
    Default: 0 """,
    )

    group = parser.add_argument_group("Learning Rate Scheduler")
    group.add_argument(
        "--lr_scheduler_type",
        type=str, default=None, dest="optimizer.lr_scheduler_type",
        help="""Learning Rate Schedule
    (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) \n

    - lambda: Sets the learning rate of each parameter group to the
        initial lr times a given function.
    - step: Sets the learning rate of each parameter group to the
        initial lr decayed by gamma every step_size epochs.
    - multi_step: Set the learning rate of each parameter group to
        the initial lr decayed by gamma once the number of epoch
        reaches one of the milestones.
    - exponential: Set the learning rate of each parameter group to
        the initial lr decayed by gamma every epoch.
    - cosine: Set the learning rate of each parameter group using
        a cosine annealing schedule, where ηmax is set to the initial
        lr and Tcur is the number of epochs since the last restart in SGDR:
        SGDR: Stochastic Gradient Descent with Warm Restarts
        (https://arxiv.org/abs/1608.03983)
    When last_epoch=-1, sets initial lr as lr.

    - reduce_on_plateau: Reduce learning rate when a metric has
        stopped improving. Models often benefit from reducing the
        learning rate by a factor of 2-10 once learning stagnates.
        This scheduler reads a metrics quantity and if no improvement
        is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    - warmup: a learning rate warm-up scheme with an inverse exponential increase
         from 0.0 to {learning_rate} in the first {final_step}.

    [step|multi_step|exponential|reduce_on_plateau|cosine|warmup]
        """,
    )

    group = parser.add_argument_group("  # StepLR")
    group.add_argument(
        "--step.step_size",
        type=int, default=1, dest="optimizer.step.step_size",
        help="""\
    Period of learning rate decay.
    Default: 1""",
    )
    group.add_argument(
        "--step.gamma",
        type=float, default=0.1, dest="optimizer.step.gamma",
        help="""\
    Multiplicative factor of learning rate decay.
    Default: 0.1. """,
    )
    group.add_argument(
        "--step.last_epoch",
        type=int, default=-1, dest="optimizer.step.last_epoch",
        help="""\
    The index of last epoch.
    Default: -1. """
    )

    group = parser.add_argument_group("  # MultiStepLR")
    group.add_argument(
        "--multi_step.milestones", nargs="+",
        type=int, dest="optimizer.multi_step.milestones",
        help="""\
    List of epoch indices. Must be increasing
    list of int""",
    )
    group.add_argument(
        "--multi_step.gamma",
        type=float, default=0.1, dest="optimizer.multi_step.gamma",
        help="""\
    Multiplicative factor of learning rate decay.
    Default: 0.1. """,
    )
    group.add_argument(
        "--multi_step.last_epoch",
        type=int, default=-1, dest="optimizer.multi_step.last_epoch",
        help="""\
    The index of last epoch.
    Default: -1. """
    )

    group = parser.add_argument_group("  # ExponentialLR")
    group.add_argument(
        "--exponential.gamma",
        type=float, default=0.1, dest="optimizer.exponential.gamma",
        help="""\
    Multiplicative factor of learning rate decay.
    Default: 0.1. """,
    )
    group.add_argument(
        "--exponential.last_epoch",
        type=int, default=-1, dest="optimizer.exponential.last_epoch",
        help="""\
    The index of last epoch.
    Default: -1. """
    )

    group = parser.add_argument_group("  # CosineAnnealingLR")
    group.add_argument(
        "--cosine.T_max",
        type=int, default=50, dest="optimizer.cosine.T_max",
        help="""\
    Maximum number of iterations.
    Default: 50""",
    )
    group.add_argument(
        "--cosine.eta_min",
        type=float, default=0, dest="optimizer.cosine.eta_min",
        help="""\
    Minimum learning rate.
    Default: 0. """,
    )
    group.add_argument(
        "--cosine.last_epoch",
        type=int, default=-1, dest="optimizer.cosine.last_epoch",
        help="""\
    The index of last epoch.
    Default: -1. """
    )

    group = parser.add_argument_group("  # ReduceLROnPlateau")
    group.add_argument(
        "--reduce_on_plateau.factor",
        type=float, default=0.1, dest="optimizer.reduce_on_plateau.factor",
        help=""" Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1. """,
    )
    group.add_argument(
        "--reduce_on_plateau.mode",
        type=str, default="min", dest="optimizer.reduce_on_plateau.mode",
        help="""\
    One of `min`, `max`. In `min` mode, lr will
    be reduced when the quantity monitored has stopped
    decreasing; in `max` mode it will be reduced when the
    quantity monitored has stopped increasing.
    Default: 'min'. """,
    )
    group.add_argument(
        "--reduce_on_plateau.patience",
        type=int, default=10, dest="optimizer.reduce_on_plateau.patience",
        help="""\
    Number of epochs with no improvement after which learning rate will be reduced.
    Default: 10. """,
    )
    group.add_argument(
        "--reduce_on_plateau.threshold",
        type=float, default=1e-4, dest="optimizer.reduce_on_plateau.threshold",
        help="""\
    Threshold for measuring the new optimum, to only focus on significant changes.
    Default: 1e-4 """,
    )
    group.add_argument(
        "--reduce_on_plateau.threshold_mode",
        type=str, default="rel", dest="optimizer.reduce_on_plateau.threshold_mode",
        help="""\
    One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or
    best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold
    in max mode or best - threshold in min mode.
    Default: ‘rel’. """
    )
    group.add_argument(
        "--reduce_on_plateau.cooldown",
        type=int, default=0, dest="optimizer.reduce_on_plateau.cooldown",
        help="""\
    Number of epochs to wait before resuming normal operation after lr has been reduced.
    Default: 0. """,
    )
    group.add_argument(
        "--reduce_on_plateau.min_lr", nargs="+",
        type=float, default=0, dest="optimizer.reduce_on_plateau.min_lr",
        help="""\
    A scalar or a list of scalars. A lower bound on the learning rate of
    all param groups or each group respectively.
    Default: 0. """,
    )
    group.add_argument(
        "--reduce_on_plateau.eps",
        type=float, default=1e-8, dest="optimizer.reduce_on_plateau.eps",
        help="""\
    Minimal decay applied to lr. If the difference between new and
    old lr is smaller than eps, the update is ignored.
    Default: 1e-8 """,
    )

    group = parser.add_argument_group("  # WarmUpLR")
    group.add_argument(
        "--warmup.final_step",
        type=int, default=1000, dest="optimizer.warmup.final_step",
        help="""\
    The number of steps to exponential increase the learning rate.
    Default: 1000. """,
    )
    group.add_argument(
        "--warmup.last_epoch",
        type=int, default=-1, dest="optimizer.warmup.last_epoch",
        help="""\
    The index of last epoch.
    Default: -1. """
    )

    group = parser.add_argument_group("Exponential Moving Average")
    group.add_argument(
        "--ema",
        type=float, default=None, dest="optimizer.exponential_moving_average",
        help="""\
    Exponential Moving Average
    Default: None (don't use)""",
    )


def base_config(parser):

    group = parser.add_argument_group("Base Config")
    group.add_argument(
        "--base_config",
        type=str, default=None, dest="base_config",
        help=f"""\
    Use pre-defined base_config:
    {_get_define_config()}

    * SQuAD:
    {_get_define_config(category='squad')}

    * KorQuAD:
    {_get_define_config(category='korquad')}

    * WikiSQL:
    {_get_define_config(category='wikisql')}

    * CoLA:
    {_get_define_config(category='cola')}

    * CoNLL 2003:
    {_get_define_config(category='conll2003')}
    """,
    )


def _get_define_config(category=None, config_dir="base_config"):
    if category is not None:
        config_dir = os.path.join(config_dir, category)

    config_files = [
        config_path.replace(".json", "")
        for config_path in os.listdir(config_dir)
        if config_path.endswith(".json")
    ]

    if category is not None:
        config_files = [category + "/" + fname for fname in config_files]
    return config_files


def evaluate(parser):

    group = parser.add_argument_group("Run evaluate")
    group.add_argument(
        "data_file_path",
        type=str,
        help=" Path to the file containing the evaluation data"
    )
    group.add_argument("checkpoint_path", type=str, help="Path to an checkpoint trained model")
    group.add_argument(
        "--infer",
        default=None, dest="inference_latency", type=int,
        help=""" Evaluate with inference-latency with maximum value (ms)""",
    )
    group.add_argument(
        "--prev_cuda_device_id",
        type=int, default=0, dest="prev_cuda_device_id",
        help=""" Previous cuda device id (need to mapping)""",
    )


def predict(parser):

    group = parser.add_argument_group("Run inference")
    group.add_argument(
        "checkpoint_path",
        type=str,
        help=" Path to an checkpoint trained model")
    group.add_argument(
        "-i", "--interactive",
        default=False, dest="interactive", action="store_true",
        help=""" Interactive Mode """,
    )
    group.add_argument(
        "--prev_cuda_device_id",
        type=int, default=0, dest="prev_cuda_device_id",
        help=""" Previous cuda device id (need to mapping)""",
    )

    group.add_argument("--question",
                       type=str, dest="question",
                       help=""" Input Question (required)""")

    group = parser.add_argument_group(" # Reading Comprehension")
    group.add_argument("--context",
                       type=str, dest="context",
                       help=""" Input Context """)

    group = parser.add_argument_group(" # Semantic Parsing")
    group.add_argument("--column", nargs="+",
                       type=str, dest="column",
                       help=""" Input Database Columns """)
    group.add_argument("--db_path",
                       type=str, dest="db_path",
                       help=""" Input Database file path """)
    group.add_argument("--table_id",
                       type=str, dest="table_id",
                       help=""" Input Database Table Id """)

    group = parser.add_argument_group(" # Document Retrieval")
    group.add_argument("--doc_path",
                       type=str, dest="doc_path",
                       help=""" Document file Path """)

    group.add_argument(
        "--retrieval",
        type=str, default=None, dest="doc_retrieval",
        help=""" Document Retrieval Model [tfidf] """,
    )
    group.add_argument("--k",
                       type=int, default=1, dest="top_k",
                       help=""" Return Top K results """)

    group = parser.add_argument_group(" # Sequence/Token Classification")
    group.add_argument("--sequence",
                       type=str, dest="sequence",
                       help=""" Input Sequence """)


def machine(parser):

    group = parser.add_argument_group("Machine Config")
    group.add_argument(
        "--machine_config",
        type=str, default=None, dest="machine_config",
        help=f"""\
    Use pre-defined machine_config (.json)

    {_get_define_config(config_dir="./machine_config")}
    """)

# fmt: on
