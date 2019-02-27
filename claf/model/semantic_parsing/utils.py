
import numpy as np
import torch


def encode_column(column_embed, column_name_mask, rnn_module):
    B, C_L, N_L, embed_D = list(column_embed.size())

    column_lengths = get_column_lengths(column_embed, column_name_mask)
    column_last_index = column_lengths - column_lengths.gt(0).long()  # NOTE: hard-code

    column_reshape = [-1] + [N_L, embed_D]
    column_embed = column_embed.view(*column_reshape)

    encoded_column, _ = rnn_module(column_embed)
    encoded_D = encoded_column.size(-1)

    encoded_output_column = torch.cat(
        [
            torch.index_select(encoded_column[i], 0, column_last_index[i])
            for i in range(column_last_index.size(0))
        ],
        dim=0,
    )
    encoded_output_column = encoded_output_column.view([B, C_L, encoded_D])
    return encoded_output_column


def get_column_lengths(column_embed, column_name_mask):
    _, _, N_L, embed_D = list(column_embed.size())
    column_reshape = [-1] + [N_L, embed_D]

    return torch.sum(column_name_mask.view(*column_reshape[:-1]), dim=-1).long()


def filter_used_column(encoded_columns, col_idx, padding_count=4):
    B, C_L, D = list(encoded_columns.size())
    zero_padding = torch.zeros(D)
    if torch.cuda.is_available():
        zero_padding = zero_padding.cuda(torch.cuda.current_device())

    encoded_used_columns = []
    for i in range(B):
        encoded_used_column = torch.stack(
            [encoded_columns[i][j] for j in col_idx[i]]
            + [zero_padding] * (padding_count - len(col_idx[i]))
        )
        encoded_used_columns.append(encoded_used_column)
    return torch.stack(encoded_used_columns)


def convert_position_to_decoder_input(conds_val_pos, token_maxlen=200):
    B = len(conds_val_pos)
    max_len = (
        max([max([len(tok) for tok in tok_seq] + [0]) for tok_seq in conds_val_pos]) - 1
    )  # The max seq len in the batch.
    if max_len < 1:
        max_len = 1
    ret_array = np.zeros((B, 4, max_len, token_maxlen), dtype=np.float32)
    ret_len = np.zeros((B, 4))
    for b, tok_seq in enumerate(conds_val_pos):
        idx = 0
        for idx, one_tok_seq in enumerate(tok_seq):
            out_one_tok_seq = one_tok_seq[:-1]
            ret_len[b, idx] = len(out_one_tok_seq)
            for t, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, idx, t, tok_id] = 1
        if idx < 3:
            ret_array[b, idx + 1 :, 0, 1] = 1
            ret_len[b, idx + 1 :] = 1

    ret_inp = torch.from_numpy(ret_array)
    if torch.cuda.is_available():
        ret_inp = ret_inp.cuda(torch.cuda.current_device())

    return ret_inp, ret_len  # [B, IDX, max_len, token_maxlen]
