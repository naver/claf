
import torch

import claf.modules.functional as f


def test_add_masked_value():
    a = torch.rand(3, 5)
    a_mask = torch.FloatTensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])

    tensor = f.add_masked_value(a, a_mask, value=100)

    assert tensor[0][3] == 100
    assert tensor[0][4] == 100
    assert tensor[1][2] == 100
    assert tensor[1][3] == 100
    assert tensor[1][4] == 100


def test_add_masked_value_with_byte_tensor():
    a = torch.rand(3, 5)
    a_mask = torch.ByteTensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])

    tensor = f.add_masked_value(a, a_mask, value=100)

    assert tensor[0][3] == 100
    assert tensor[0][4] == 100
    assert tensor[1][2] == 100
    assert tensor[1][3] == 100
    assert tensor[1][4] == 100


def test_get_mask_from_tokens_with_2_dim():
    tokens = {
        "word" : torch.LongTensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]),
    }

    mask = f.get_mask_from_tokens(tokens)
    print(mask)
    assert mask.equal(tokens["word"])


def test_get_mask_from_tokens_with_3_dim():
    tokens = {
        "char" : torch.LongTensor([
            [[4, 2], [3, 6], [0, 0]],
            [[5, 1], [0, 0], [0, 0]],
            [[1, 3], [2, 4], [3, 6]],
        ]),
    }

    mask = f.get_mask_from_tokens(tokens)
    expect_tensor = torch.LongTensor([
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ])
    assert mask.equal(expect_tensor)


def test_last_dim_masked_softmax_with_2_dim():
    tensor = torch.FloatTensor([
            [2, 3, 1, 0, 0],
            [4, 1, 0, 0, 0],
            [1, 5, 2, 4, 1],
        ])
    mask = f.get_mask_from_tokens({"word": tensor}).float()

    result = f.last_dim_masked_softmax(tensor, mask)
    assert result.argmax(dim=-1).equal(torch.LongTensor([1, 0, 1]))


def test_masked_softmax():
    tensor = torch.FloatTensor([
            [2, 3, 1, 4, 5],
            [4, 1, 6, 9, 10],
            [1, 5, 2, 4, 1],
        ])
    mask = torch.tensor([
        [1., 1., 1., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1.]
    ])

    result = f.masked_softmax(tensor, mask)
    assert result.argmax(dim=-1).equal(torch.LongTensor([1, 0, 1]))


def test_masked_zero():
    tensor = torch.FloatTensor([
            [2, 3, 1, 4, 5],
            [4, 1, 6, 9, 10],
            [1, 5, 2, 4, 1],
        ])
    mask = torch.tensor([
        [1., 1., 1., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1.]
    ])

    result = f.masked_zero(tensor, mask)
    assert result[0][3] == 0
    assert result[0][4] == 0
    assert result[1][2] == 0
    assert result[1][3] == 0
    assert result[1][4] == 0

    result = f.masked_zero(tensor.long(), mask)
    assert result[0][3] == 0
    assert result[0][4] == 0
    assert result[1][2] == 0
    assert result[1][3] == 0
    assert result[1][4] == 0

    result = f.masked_zero(tensor.byte(), mask)
    assert result[0][3] == 0
    assert result[0][4] == 0
    assert result[1][2] == 0
    assert result[1][3] == 0
    assert result[1][4] == 0


def test_get_sorted_seq_config():
    tensor = torch.LongTensor([
            [2, 3, 1, 0, 0],
            [4, 1, 0, 0, 0],
            [1, 5, 2, 4, 1],
        ])

    seq_config = f.get_sorted_seq_config({"word": tensor})
    assert seq_config["seq_lengths"].tolist() == [5, 3, 2]
    assert seq_config["perm_idx"].tolist() == [2, 0, 1]
    assert seq_config["unperm_idx"].tolist() == [1, 2, 0]


def test_forward_rnn_with_pack():
    tensor = torch.LongTensor([
            [2, 3, 1, 0, 0],
            [4, 1, 0, 0, 0],
            [1, 5, 2, 4, 1],
        ])
    matrix = torch.rand(10, 10)
    embedded_tensor = torch.nn.functional.embedding(tensor, matrix)

    seq_config = f.get_sorted_seq_config({"word": tensor})

    gru = torch.nn.GRU(input_size=10, hidden_size=1, bidirectional=False, batch_first=True)
    encoded_tensor = f.forward_rnn_with_pack(gru, embedded_tensor, seq_config)
    assert encoded_tensor[0][3] == 0
    assert encoded_tensor[0][4] == 0
    assert encoded_tensor[1][2] == 0
    assert encoded_tensor[1][3] == 0
    assert encoded_tensor[1][4] == 0
