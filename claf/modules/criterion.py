
import torch
import torch.nn as nn


def get_criterion_fn(name, **kwargs):
    ignore_index = kwargs["ignore_index"] if "ignore_index" in kwargs else -100
    label_smoothing = kwargs["label_smoothing"] if "label_smoothing" in kwargs else 0.
    crf = kwargs["crf"] if "crf" in kwargs else None

    def cross_entropy(logits, targets):
        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        return cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

    def cross_entropy_with_label_smoothing(logits, targets):
        assert label_smoothing > 0.
        return softmax_cross_entropy_with_logits(
            logits, targets, label_smoothing=label_smoothing, ignore_index=ignore_index
        )

    def crf_negative_log_likelihood(logits, targets):
        assert crf is not None

        return _crf_negative_log_likelihood(
            logits, targets, ignore_index, crf
        )

    criterion_functions = {
        "cross_entropy": cross_entropy,
        "cross_entropy_with_label_smoothing": cross_entropy_with_label_smoothing,
        "crf_negative_log_likelihood": crf_negative_log_likelihood,
    }

    if name not in criterion_functions:
        raise ValueError(
            f"'{name}' is not included in criterion_functions. use below one. \n {criterion_functions.keys()}"
        )

    return criterion_functions[name]


def _crf_negative_log_likelihood(logits, targets, ignore_index, crf):
    mask = (targets != ignore_index).long()
    targets[targets == ignore_index] = 0
    log_likelihood = crf(logits, targets, mask)
    return -log_likelihood


def get_label_smoothed_targets(logits, targets, label_smoothing, ignore_index):
    num_classes = logits.size(-1)
    smoothing_value = label_smoothing / num_classes

    smoothed_targets = get_one_hot_targets(logits, targets, ignore_index)
    real_positions = (smoothed_targets.sum(dim=-1, keepdim=True) != 0).float()

    smoothed_targets[smoothed_targets == 1.0] -= label_smoothing
    smoothed_targets = smoothed_targets + smoothing_value

    return smoothed_targets * real_positions


def get_one_hot_targets(logits, targets, ignore_index):
    orig_shape = logits.size()
    num_classes = orig_shape[-1]

    real_logit_positions = (targets != ignore_index).unsqueeze(-1).float()  # [B, L, 1]

    one_hot = torch.zeros_like(logits)
    one_hot_flat = one_hot.reshape(-1, num_classes)

    targets[targets == ignore_index] = 0
    one_hot_flat = one_hot_flat.scatter(-1, targets.reshape(-1, 1), 1.0)  # [B, L, D]
    one_hot = one_hot_flat.reshape(*orig_shape) * real_logit_positions

    return one_hot


def softmax_cross_entropy_with_logits(logits, targets, label_smoothing=0.0, ignore_index=-1):
    N = (targets != ignore_index).view(-1).sum().float()

    if not label_smoothing:
        targets_prob = get_one_hot_targets(logits, targets, ignore_index)
    else:
        targets_prob = get_label_smoothed_targets(logits, targets, label_smoothing, ignore_index)

    return torch.sum(-targets_prob * torch.nn.functional.log_softmax(logits, dim=-1)) / N
