
from overrides import overrides

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from claf.decorator import register
from claf.model.base import ModelWithTokenEmbedder
from claf.model.semantic_parsing import utils
from claf.model.semantic_parsing.mixin import WikiSQL
import claf.modules.functional as f
import claf.modules.attention as attention


@register("model:sqlnet")
class SQLNet(WikiSQL, ModelWithTokenEmbedder):
    """
    Nature Language to SQL Query Model. `Semantic Parsing`, `NL2SQL`

    Implementation of model presented in
    SQLNet: Generating Structured Queries From Natural Language
      Without Reinforcement Learning
    (https://arxiv.org/abs/1711.04436)

    * Args:
        token_embedder: 'WikiSQLTokenEmbedder', Used to embed the 'column' and 'question'.

    * Kwargs:
        column_attention: highlight that column attention is a special instance of
          the generic attention mechanism to compute the attention map on a question
          conditioned on the column names.
        model_dim: the number of model dimension
        rnn_num_layer: the number of recurrent layers (all of rnn)
        column_maxlen: an upper-bound N on the number of columns to choose
        token_maxlen: conds value slot - pointer network an upper-bound N on the number of token
        conds_column_loss_alpha: balance the positive data versus negative data
    """

    def __init__(
        self,
        token_embedder,
        column_attention=None,
        model_dim=100,
        rnn_num_layer=2,
        dropout=0.3,
        column_maxlen=4,
        token_maxlen=200,
        conds_column_loss_alpha=3,
    ):
        super(SQLNet, self).__init__(token_embedder)

        embed_dim = token_embedder.get_embed_dim()  # NOTE: need to fix
        self.token_maxlen = token_maxlen
        self.column_maxlen = column_maxlen
        self.conds_column_loss_alpha = conds_column_loss_alpha

        # Predict aggregator
        self.agg_predictor = AggPredictor(
            embed_dim, model_dim, rnn_num_layer, dropout, len(self.AGG_OPS)
        )

        # Predict selected column
        self.sel_predictor = SelPredictor(
            embed_dim, model_dim, rnn_num_layer, dropout, column_attention=column_attention
        )

        # #Predict number of conditions
        self.conds_predictor = CondsPredictor(
            embed_dim,
            model_dim,
            rnn_num_layer,
            dropout,
            len(self.COND_OPS),
            column_maxlen,
            token_maxlen,
            column_attention=column_attention,
        )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()

    @overrides
    def forward(self, features, labels=None):
        column = features["column"]
        question = features["question"]

        column_embed = self.token_embedder(column)
        question_embed = self.token_embedder(question)

        B, C_L = column_embed.size(0), column_embed.size(1)

        column_indexed = column[next(iter(column))]
        column_name_mask = column_indexed.gt(0).float()  # NOTE: hard-code
        column_lengths = utils.get_column_lengths(column_embed, column_name_mask)
        column_mask = column_lengths.view(B, C_L).gt(0).float()  # NOTE: hard-code
        question_mask = f.get_mask_from_tokens(question).float()

        agg_logits = self.agg_predictor(question_embed, question_mask)
        sel_logits = self.sel_predictor(
            question_embed, question_mask, column_embed, column_name_mask, column_mask
        )

        conds_col_idx, conds_val_pos = None, None
        if labels:
            data_idx = labels["data_idx"]
            ground_truths = self._dataset.get_ground_truths(data_idx)

            conds_col_idx = [ground_truth["conds_col"] for ground_truth in ground_truths]
            conds_val_pos = [ground_truth["conds_val_pos"] for ground_truth in ground_truths]

        conds_logits = self.conds_predictor(
            question_embed,
            question_mask,
            column_embed,
            column_name_mask,
            column_mask,
            conds_col_idx,
            conds_val_pos,
        )

        # Convert GPU to CPU
        agg_logits = agg_logits.cpu()
        sel_logits = sel_logits.cpu()
        conds_logits = [logits.cpu() for logits in conds_logits]

        output_dict = {
            "agg_logits": agg_logits,
            "sel_logits": sel_logits,
            "conds_logits": conds_logits,
        }

        if labels:
            data_idx = labels["data_idx"]
            output_dict["data_id"] = data_idx

            ground_truths = self._dataset.get_ground_truths(data_idx)

            # Aggregator, Select Column
            target_agg_idx = torch.LongTensor(
                [ground_truth["agg_idx"] for ground_truth in ground_truths]
            )
            target_sel_idx = torch.LongTensor(
                [ground_truth["sel_idx"] for ground_truth in ground_truths]
            )

            loss = 0
            loss += self.cross_entropy(agg_logits, target_agg_idx)
            loss += self.cross_entropy(sel_logits, target_sel_idx)

            conds_num_logits, conds_column_logits, conds_op_logits, conds_value_logits = (
                conds_logits
            )

            # Conditions
            # 1. The number of conditions
            target_conds_num = torch.LongTensor(
                [ground_truth["conds_num"] for ground_truth in ground_truths]
            )
            target_conds_column = [ground_truth["conds_col"] for ground_truth in ground_truths]

            loss += self.cross_entropy(conds_num_logits, target_conds_num)

            # 2. Columns of conditions
            B = conds_column_logits.size(0)

            target_conds_columns = np.zeros(list(conds_column_logits.size()), dtype=np.float32)
            for i in range(B):
                target_conds_column_idx = target_conds_column[i]
                if len(target_conds_column_idx) == 0:
                    continue
                target_conds_columns[i][target_conds_column_idx] = 1
            target_conds_columns = torch.from_numpy(target_conds_columns)
            conds_column_probs = torch.sigmoid(conds_column_logits)

            bce_loss = -torch.mean(
                self.conds_column_loss_alpha
                * (target_conds_columns * torch.log(conds_column_probs + 1e-10))
                + (1 - target_conds_columns) * torch.log(1 - conds_column_probs + 1e-10)
            )
            loss += bce_loss

            # 3. Operator of conditions
            conds_op_loss = 0
            for i in range(B):
                target_conds_op = ground_truths[i]["conds_op"]
                if len(target_conds_op) == 0:
                    continue

                target_conds_op = torch.from_numpy(np.array(target_conds_op))
                logits_conds_op = conds_op_logits[i, : len(target_conds_op)]

                target_op_count = len(target_conds_op)
                conds_op_loss += (
                    self.cross_entropy(logits_conds_op, target_conds_op) / target_op_count
                )
            loss += conds_op_loss

            # 4. Value of conditions
            conds_val_pos = [ground_truth["conds_val_pos"] for ground_truth in ground_truths]

            conds_value_loss = 0
            for i in range(B):
                for j in range(len(conds_val_pos[i])):
                    cond_val_pos = conds_val_pos[i][j]
                    if len(cond_val_pos) == 1:
                        continue

                    target_cond_val_pos = torch.from_numpy(
                        np.array(cond_val_pos[1:])
                    )  # index 0: START_TOKEN
                    logits_cond_val_pos = conds_value_logits[i, j, : len(cond_val_pos) - 1]

                    conds_value_loss += self.cross_entropy(
                        logits_cond_val_pos, target_cond_val_pos
                    ) / len(conds_val_pos[i])

            loss += conds_value_loss / B

            output_dict["loss"] = loss.unsqueeze(0)

        return output_dict


class AggPredictor(nn.Module):
    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, agg_count):
        super(AggPredictor, self).__init__()

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, agg_count)
        )

    def forward(self, question_embed, question_mask):
        encoded_question, _ = self.question_rnn(question_embed)
        attn_matrix = self.seq_attn(encoded_question, question_mask)
        attn_question = f.weighted_sum(attn_matrix, encoded_question)
        logits = self.mlp(attn_question)
        return logits


class SelPredictor(nn.Module):
    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, column_attention=None):
        super(SelPredictor, self).__init__()
        self.column_attention = column_attention

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)

        self.column_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(model_dim, 1))

    def forward(self, question_embed, question_mask, column_embed, column_name_mask, column_mask):

        B, C_L, N_L, embed_D = list(column_embed.size())

        encoded_column = utils.encode_column(column_embed, column_name_mask, self.column_rnn)
        encoded_question, _ = self.question_rnn(question_embed)

        if self.column_attention:
            attn_matrix = torch.bmm(
                encoded_column, self.linear_attn(encoded_question).transpose(1, 2)
            )
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.unsqueeze(1), value=-1e7)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)

        logits = self.mlp(
            self.linear_question(attn_question) + self.linear_column(encoded_column)
        ).squeeze()
        logits = f.add_masked_value(logits, column_mask, value=-1e7)
        return logits


class CondsPredictor(nn.Module):
    def __init__(
        self,
        embed_dim,
        model_dim,
        rnn_num_layer,
        dropout,
        conds_op_count,
        column_maxlen,
        token_maxlen,
        column_attention=None,
    ):
        super(CondsPredictor, self).__init__()

        self.num_predictor = CondsNumPredictor(
            embed_dim, model_dim, rnn_num_layer, dropout, column_maxlen
        )
        self.column_predictor = CondsColPredictor(
            embed_dim, model_dim, rnn_num_layer, dropout, column_attention=column_attention
        )
        self.op_predictor = CondsOpPredictor(
            embed_dim,
            model_dim,
            rnn_num_layer,
            dropout,
            conds_op_count,
            column_maxlen,
            column_attention=column_attention,
        )
        self.value_pointer = CondsValuePointer(
            embed_dim, model_dim, rnn_num_layer, dropout, column_maxlen, token_maxlen
        )

    def forward(
        self,
        question_embed,
        question_mask,
        column_embed,
        column_name_mask,
        column_mask,
        col_idx,
        conds_val_pos,
    ):
        num_logits = self.num_predictor(
            question_embed, question_mask, column_embed, column_name_mask, column_mask
        )
        column_logits = self.column_predictor(
            question_embed, question_mask, column_embed, column_name_mask, column_mask
        )

        if col_idx is None:
            col_idx = []
            preds_num = torch.argmax(num_logits, dim=-1)
            for i in range(column_logits.size(0)):
                _, pred_conds_column_idx = torch.topk(column_logits[i], preds_num[i])
                col_idx.append(pred_conds_column_idx.tolist())

        op_logits = self.op_predictor(
            question_embed, question_mask, column_embed, column_name_mask, col_idx
        )
        value_logits = self.value_pointer(
            question_embed, question_mask, column_embed, column_name_mask, col_idx, conds_val_pos
        )

        return (num_logits, column_logits, op_logits, value_logits)


class CondsNumPredictor(nn.Module):
    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, column_maxlen):
        super(CondsNumPredictor, self).__init__()

        self.model_dim = model_dim
        self.column_maxlen = column_maxlen

        self.column_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.column_seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_to_hidden_state = nn.Linear(model_dim, 2 * model_dim)
        self.column_to_cell_state = nn.Linear(model_dim, 2 * model_dim)

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.question_seq_attn = attention.LinearSeqAttn(model_dim)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, column_maxlen + 1)
        )

    def forward(self, question_embed, question_mask, column_embed, column_name_mask, column_mask):
        B, C_L, N_L, embed_D = list(column_embed.size())

        encoded_column = utils.encode_column(column_embed, column_name_mask, self.column_rnn)
        attn_column = self.column_seq_attn(encoded_column, column_mask)
        out_column = f.weighted_sum(attn_column, encoded_column)

        question_rnn_hidden_state = (
            self.column_to_hidden_state(out_column)
            .view(B, self.column_maxlen, self.model_dim // 2)
            .transpose(0, 1)
            .contiguous()
        )
        question_rnn_cell_state = (
            self.column_to_cell_state(out_column)
            .view(B, self.column_maxlen, self.model_dim // 2)
            .transpose(0, 1)
            .contiguous()
        )

        encoded_question, _ = self.question_rnn(
            question_embed, (question_rnn_hidden_state, question_rnn_cell_state)
        )
        attn_question = self.question_seq_attn(encoded_question, question_mask)
        out_question = f.weighted_sum(attn_question, encoded_question)
        return self.mlp(out_question)


class CondsColPredictor(nn.Module):
    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, column_attention=None):
        super(CondsColPredictor, self).__init__()
        self.column_attention = column_attention

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)

        self.column_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, question_embed, question_mask, column_embed, column_name_mask, column_mask):
        B, C_L, N_L, embed_D = list(column_embed.size())

        # Column Encoder
        encoded_column = utils.encode_column(column_embed, column_name_mask, self.column_rnn)
        encoded_question, _ = self.question_rnn(question_embed)

        if self.column_attention:
            attn_matrix = torch.bmm(
                encoded_column, self.linear_attn(encoded_question).transpose(1, 2)
            )
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.unsqueeze(1), value=-1e7)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)

        logits = self.mlp(
            self.linear_question(attn_question) + self.linear_column(encoded_column)
        ).squeeze()
        logits = f.add_masked_value(logits, column_mask, value=-1e7)
        return logits


class CondsOpPredictor(nn.Module):
    def __init__(
        self,
        embed_dim,
        model_dim,
        rnn_num_layer,
        dropout,
        op_count,
        column_maxlen,
        column_attention=None,
    ):
        super(CondsOpPredictor, self).__init__()
        self.column_attention = column_attention
        self.column_maxlen = column_maxlen

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)

        self.column_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.Tanh(), nn.Linear(model_dim, op_count)
        )

    def forward(self, question_embed, question_mask, column_embed, column_name_mask, col_idx):
        B, C_L, N_L, embed_D = list(column_embed.size())

        # Column Encoder
        encoded_column = utils.encode_column(column_embed, column_name_mask, self.column_rnn)
        encoded_used_column = utils.filter_used_column(
            encoded_column, col_idx, padding_count=self.column_maxlen
        )

        encoded_question, _ = self.question_rnn(question_embed)
        if self.column_attention:
            attn_matrix = torch.matmul(
                self.linear_attn(encoded_question).unsqueeze(1), encoded_used_column.unsqueeze(3)
            ).squeeze()
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.unsqueeze(1), value=-1e7)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)

        return self.mlp(
            self.linear_question(attn_question) + self.linear_column(encoded_used_column)
        ).squeeze()


class CondsValuePointer(nn.Module):
    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, column_maxlen, token_maxlen):
        super(CondsValuePointer, self).__init__()

        self.model_dim = model_dim
        self.column_maxlen = column_maxlen
        self.token_maxlen = token_maxlen

        self.question_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.seq_attn = attention.LinearSeqAttn(model_dim)

        self.column_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=model_dim // 2,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.decoder = nn.LSTM(
            input_size=self.token_maxlen,
            hidden_size=model_dim,
            num_layers=rnn_num_layer,
            batch_first=True,
            dropout=dropout,
        )

        self.linear_column = nn.Linear(model_dim, model_dim)
        self.linear_conds = nn.Linear(model_dim, model_dim)
        self.linear_question = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(
        self, question_embed, question_mask, column_embed, column_name_mask, col_idx, conds_val_pos
    ):
        B, C_L, N_L, embed_D = list(column_embed.size())

        question_embed, question_mask = self.concat_start_and_end_zero_padding(
            question_embed, question_mask
        )

        # Column Encoder
        encoded_column = utils.encode_column(column_embed, column_name_mask, self.column_rnn)
        encoded_used_column = utils.filter_used_column(
            encoded_column, col_idx, padding_count=self.column_maxlen
        )

        encoded_question, _ = self.question_rnn(question_embed)

        encoded_used_column = encoded_used_column.unsqueeze(2).unsqueeze(2)
        encoded_question = encoded_question.unsqueeze(1).unsqueeze(1)

        if conds_val_pos is None:  # inference
            MAX_DECODER_STEP = 50

            decoder_input = torch.zeros(4 * B, 1, self.token_maxlen)
            decoder_input[:, 0, 0] = 2  # Set <s> Token
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            decoder_hidden = None

            logits = []
            for _ in range(MAX_DECODER_STEP):
                step_logit, decoder_hidden = self.decode_then_output(
                    encoded_used_column,
                    encoded_question,
                    question_mask,
                    decoder_input,
                    decoder_hidden=decoder_hidden,
                )
                step_logit = step_logit.unsqueeze(1)
                logits.append(step_logit)

                # To ont-hot
                _, decoder_idxs = step_logit.view(B * self.column_maxlen, -1).max(1)
                decoder_input = torch.zeros(B * self.column_maxlen, self.token_maxlen).scatter_(
                    1, decoder_idxs.cpu().unsqueeze(1), 1
                )
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()

            logits = torch.stack(logits, 2)
        else:
            decoder_input, _ = utils.convert_position_to_decoder_input(
                conds_val_pos, token_maxlen=self.token_maxlen
            )
            logits, _ = self.decode_then_output(
                encoded_used_column, encoded_question, question_mask, decoder_input
            )
        return logits

    def concat_start_and_end_zero_padding(self, question_embed, mask):
        B, Q_L, embed_D = list(question_embed.size())

        zero_padding = torch.zeros(B, 1, embed_D)
        mask_with_start_end = torch.zeros(B, Q_L + 2)

        if torch.cuda.is_available():
            zero_padding = zero_padding.cuda(torch.cuda.current_device())
            mask_with_start_end = mask_with_start_end.cuda(torch.cuda.current_device())

        question_embed_with_start_end = torch.cat(
            [zero_padding, question_embed, zero_padding], dim=1
        )  # add <BEG> and <END>

        mask_with_start_end[:, 0] = 1  # <BEG>
        mask_with_start_end[:, 1 : Q_L + 1] = mask
        question_lengths = torch.sum(mask, dim=-1).byte()
        for i in range(B):
            mask_with_start_end[i, question_lengths[i].item() + 1] = 1  # <END>

        return question_embed_with_start_end, mask_with_start_end

    def decode_then_output(
        self,
        encoded_used_column,
        encoded_question,
        question_mask,
        decoder_input,
        decoder_hidden=None,
    ):
        B = encoded_used_column.size(0)

        decoder_output, decoder_hidden = self.decoder(
            decoder_input.view(B * self.column_maxlen, -1, self.token_maxlen), decoder_hidden
        )
        decoder_output = decoder_output.contiguous().view(B, self.column_maxlen, -1, self.model_dim)
        decoder_output = decoder_output.unsqueeze(3)

        logits = self.mlp(
            self.linear_column(encoded_used_column)
            + self.linear_conds(decoder_output)
            + self.linear_question(encoded_question)
        ).squeeze()
        logits = f.add_masked_value(logits, question_mask.unsqueeze(1).unsqueeze(1), value=-1e7)
        return logits, decoder_hidden
