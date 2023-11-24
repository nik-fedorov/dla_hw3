import torch
import torch.nn as nn

from tts.base.base_model import BaseModel
from tts.utils.util import get_non_pad_mask
from .transformer import Encoder, Decoder
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(BaseModel):
    def __init__(self, stats, **model_config):
        super().__init__()

        self.pad_id = model_config['pad_id']

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config, stats)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['hidden_size'], model_config['n_mels'])

    def forward(self,
                text_encoded,
                src_pos,
                mel_pos=None,
                pitch_target=None,
                energy_target=None,
                duration_target=None,
                use_target_duration=True,
                **batch):

        encoder_output = self.encoder(text_encoded, src_pos)

        (
            va_output,
            log_duration_pred,
            duration_pred_rounded,
            pitch_pred,
            energy_pred,
            mel_pos
        ) = self.variance_adaptor(encoder_output, src_pos, mel_pos, pitch_target,
                                  energy_target, duration_target, use_target_duration)

        decoder_output = self.decoder(va_output, mel_pos)

        mel_pred = self.mel_linear(decoder_output)

        mel_non_pad_mask = get_non_pad_mask(mel_pos, self.pad_id)
        return {
            'mel_pred': mel_pred * mel_non_pad_mask,
            'log_duration_pred': log_duration_pred * src_pos.ne(self.pad_id),
            'duration_pred_rounded': duration_pred_rounded * src_pos.ne(self.pad_id),
            'pitch_pred': pitch_pred * mel_non_pad_mask.squeeze(-1),
            'energy_pred': energy_pred * mel_non_pad_mask.squeeze(-1),
            'src_pad_mask': src_pos.eq(self.pad_id)
        }
