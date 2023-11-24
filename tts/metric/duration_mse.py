import torch
import torch.nn.functional as F

from tts.base.base_metric import BaseMetric


class DurationMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, log_duration_pred, duration_target, src_pad_mask, **batch):
        log_duration_target = torch.log(duration_target.float() + 1.0)
        return F.mse_loss(log_duration_pred, log_duration_target)
