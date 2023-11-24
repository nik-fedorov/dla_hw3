import torch.nn.functional as F

from tts.base.base_metric import BaseMetric


class EnergyMSE(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, energy_pred, energy_target, **batch):
        return F.mse_loss(energy_pred, energy_target)
