
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
from collections import OrderedDict

import torch


def convert_checkpoint_to_bert_model(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_weights = checkpoint["weights"]

    bert_model_weights = OrderedDict()
    for key, tensor in model_weights.items():
        if "_model" in key or "shared_layers" in key:
            new_key = key.replace("_model", "bert").replace("shared_layers", "bert")
            bert_model_weights[new_key] = tensor

    torch.save(bert_model_weights, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str,
                        help="""CLaF Checkpoint Path""")
    parser.add_argument('output_path', type=str,
                        help="""BERT model output_path""")
    args = parser.parse_args()

    convert_checkpoint_to_bert_model(args.checkpoint_path, args.output_path)
