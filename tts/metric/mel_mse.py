import torch.nn.functional as F

from tts.base.base_metric import BaseMetric


class MelMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, mel_pred, mel_target, **batch):
        return F.mse_loss(mel_pred, mel_target)
