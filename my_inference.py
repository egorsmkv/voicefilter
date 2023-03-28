import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn

from models.voicefilter.model import VoiceFilter
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.generic_utils import load_config

# config
USE_CUDA = False

# speaker_encoder parameters 
num_mels = 40
n_fft = 512
emb_dim = 256
lstm_hidden = 768
lstm_layers = 3
window = 80
stride = 40

device = torch.device('cuda' if USE_CUDA else 'cpu')

checkpoint_embedder_path = 'embedder.pt'
checkpoint_path = 'best_checkpoint.pt'

noise_utterance = '/home/yehor/5.wav'
output_path = '/home/yehor/5-denoised.wav'


class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeakerEncoder(nn.Module):
    def __init__(self, num_mels, lstm_layers, lstm_hidden, window, stride):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(num_mels, lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(lstm_hidden, emb_dim)
        self.num_mels = num_mels
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.window = window
        self.stride = stride

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.window, self.stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x


def get_embedding(embedder, ap, wave_file_path):
    emb_wav, _ = librosa.load(wave_file_path, sr=16_000)

    if USE_CUDA:
        mel = torch.from_numpy(ap.get_mel(emb_wav)).float().cuda()
        file_embedding = embedder(mel)
    else:
        mel = torch.from_numpy(ap.get_mel(emb_wav)).float()
        file_embedding = embedder(mel) #.cpu().detach().numpy()

    return file_embedding


# extract caracteristics
def normalise_and_extract_features(_embedder, _ap, _ap_embedder, mixed_path):
    # mixed_path_norm = mixed_path.replace('.wav','-norm.wav') 
    mixed_path_norm = mixed_path

    # load wavs
    mixed_wav = _ap.load_wav(mixed_path_norm)
  
    # normalise wavs
    norm_factor = np.max(np.abs(mixed_wav)) * 1.1
    mixed_wav = mixed_wav/norm_factor

    # save this is necessary for demo
    sf.write(mixed_path_norm, mixed_wav, 16_000, 'PCM_16')

    embedding = get_embedding(_embedder, _ap_embedder, mixed_path_norm) # emb_ref_path_norm
    mixed_spec, mixed_phase = _ap.get_spec_from_audio(mixed_wav, return_phase=True)

    return embedding, mixed_spec, mixed_phase, mixed_wav


def predict(_model, _embedder, _ap, _ap_embedder, mixed_path, outpath='predict.wav'):
    embedding, mixed_spec, mixed_phase, target_wav = normalise_and_extract_features(_embedder, _ap, _ap_embedder, mixed_path)
  
    # use the model
    mixed_spec = torch.from_numpy(mixed_spec).float()

    # append 1 dimension on mixed, its need because the model (ex)spected batch
    mixed_spec = mixed_spec.unsqueeze(0)
    embedding = embedding.unsqueeze(0)

    if USE_CUDA:
        embedding = embedding.cuda()
        mixed_spec = mixed_spec.cuda()

    mask = _model(mixed_spec, embedding)
    output = mixed_spec * mask

    # inverse spectogram to wav
    est_mag = output[0].cpu().detach().numpy()
    # use phase from mixed wav for reconstruct the wave
    est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)

    sf.write(outpath, est_wav, 16_000, 'PCM_16')

    return est_wav, target_wav


embedder = SpeakerEncoder(num_mels, lstm_layers, lstm_hidden, window, stride)
chkpt_embed = torch.load(checkpoint_embedder_path, map_location=device)

embedder.load_state_dict(chkpt_embed)

embedder.eval()
if USE_CUDA:
    embedder.cuda()


# load ap compativel with speaker encoder
embedder_config = {
    "backend": "voicefilter",
    "mel_spec": False,
    "audio_len": 3, 
    "voicefilter": {
        "n_fft": 1200,
        "num_mels":40,
        "num_freq": 601,
        "sample_rate": 16000,
        "hop_length": 160,
        "win_length": 400,
        "min_level_db": -100.0,
        "ref_level_db": 20.0,
        "preemphasis": 0.97,
        "power": 1.5,
        "griffin_lim_iters": 60,
    }
}
ap_embedder = AudioProcessor(embedder_config)

# load checkpoint 
checkpoint = torch.load(checkpoint_path, map_location=device)
c = load_config('config.json')

ap = AudioProcessor(c.audio) # create AudioProcessor for model
c.model_name = 'voicefilter'
model_name = c.model_name

# load model
model = VoiceFilter(c)
model.load_state_dict(checkpoint['model'])

if USE_CUDA:
    model = model.cuda()

# Do inference
est_wav, target_wav = predict(
    model, embedder, ap, ap_embedder,
    noise_utterance, outpath=output_path)

print('Done')
