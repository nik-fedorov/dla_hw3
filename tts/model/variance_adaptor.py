import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.utils.util import create_alignment, get_non_pad_mask


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config['hidden_size']
        self.filter_size = model_config['variance_predictor_filter_size']
        self.dropout = model_config['variance_predictor_dropout']

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(self.input_size, self.filter_size, kernel_size=3, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=3, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.filter_size, 1)

    def forward(self, x):
        x = self.conv_net(x)
        out = self.linear_layer(x)
        out = out.squeeze(-1)
        return out


class LengthRegulator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.pad_id = model_config['pad_id']
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        # expand_max_len = torch.max(
        #     torch.sum(duration_predictor_output, -1), -1)[0]
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1)).item()
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, src_pos, alpha=1.0, duration_target=None, use_target_duration=True):
        duration_predictor_output = self.duration_predictor(x)

        duration_rounded = (torch.exp(duration_predictor_output) - 1) * alpha
        duration_rounded = torch.round(torch.clamp(duration_rounded, min=0)).int()

        if use_target_duration:
            assert duration_target is not None
            x = self.LR(x, duration_target)
        else:
            x = self.LR(x, duration_rounded)
        return duration_predictor_output, duration_rounded, x


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config, stats):
        super().__init__()

        self.pad_id = model_config['pad_id']
        self.use_target_pitch_and_energy = model_config.get('use_target_pitch_and_energy', True)

        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.d_control = model_config.get('duration_control', 1.0)
        self.p_control = model_config.get('pitch_control', 1.0)
        self.e_control = model_config.get('energy_control', 1.0)

        n_bins = model_config["n_bins"]
        pitch_min = stats["pitch_min"]
        pitch_max = stats["pitch_max"]
        energy_min = stats["energy_min"]
        energy_max = stats["energy_max"]

        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1),
            requires_grad=False,
        )

        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(n_bins, model_config["hidden_size"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["hidden_size"])

    def get_pitch_embedding(self, x, target, mask):
        prediction = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * self.p_control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding * mask

    def get_energy_embedding(self, x, target, mask):
        prediction = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * self.e_control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding * mask

    def forward(
        self,
        x,
        src_pos,
        mel_pos,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        use_target_duration=True
    ):

        (
            log_duration_prediction,
            duration_rounded,
            x
        ) = self.length_regulator(x, src_pos, self.d_control, duration_target, use_target_duration)

        if not use_target_duration:
            # recalculate mel_pos
            lengths = duration_rounded.sum(1)
            mel_pos = torch.full((len(x), max(lengths)), self.pad_id, dtype=torch.int, device=lengths.device)
            for i, length in enumerate(lengths):
                mel_pos[i, :length] = torch.arange(1, length + 1, dtype=torch.int, device=lengths.device)

        mask = get_non_pad_mask(mel_pos, self.pad_id)
        if use_target_duration:
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mask)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mask)
        else:
            # pass None target to use predicted pitch and energy
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, None, mask)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, None, mask)
        x = x + pitch_embedding + energy_embedding

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            pitch_prediction,
            energy_prediction,
            mel_pos
        )
