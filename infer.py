# --coding:utf-8--
import os

import torchaudio
import torch
from decoder.pretrained import WavTokenizer

from huggingface_hub import hf_hub_download


device1=torch.device('cpu')

config_path = "configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = hf_hub_download(repo_id="novateur/WavTokenizer-large-speech-75token", filename="wavtokenizer_large_speech_320_v2.ckpt")
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)

input_wav = "original.wav"
output_wav = "generated.wav"

wav, sr = torchaudio.load(input_wav)
bandwidth_id = torch.tensor([0])
wav=wav.to(device1)

features, discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

bandwidth_id = torch.tensor([0])
bandwidth_id = bandwidth_id.to(device1) 

audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)   

print(f"==== All done, output audio shape: {audio_out.shape}")

torchaudio.save(output_wav, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)





