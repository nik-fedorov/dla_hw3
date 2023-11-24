import torch.nn.functional as F

from tts.base.base_metric import BaseMetric


class PitchMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, pitch_pred, pitch_target, **batch):
        return F.mse_loss(pitch_pred, pitch_target)
