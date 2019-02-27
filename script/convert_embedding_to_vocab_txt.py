
import argparse


def read_embedding_vocabs(file_path):
    print("Reading vocabs from file")
    vocabs = []
    with open(file_path, "rb") as embeddings_file:
        for line in embeddings_file:
            fields = line.decode("utf-8").rstrip().split(" ")
            word = fields[0]
            vocabs.append(word)
    return vocabs


def write_vocab(embedding_vocabs, output_path):
    print("Write vocabs")
    vocab_texts = "\n".join(embedding_vocabs)
    with open(output_path, "wb") as vocab_file:
        vocab_file.write(vocab_texts.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('embed_path', type=str,
                        help='Pretrained embedding txt path')
    parser.add_argument('output_path', type=str,
                        help='vocab_texts output path')
    args = parser.parse_args()

    embedding_vocabs = read_embedding_vocabs(args.embed_path)
    write_vocab(embedding_vocabs, args.output_path)
