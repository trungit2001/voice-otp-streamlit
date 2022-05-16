import random
import torchaudio
import numpy as np

from pickle import load
from keras import models
from torchaudio import transforms as T
from torchaudio import functional as F
from models.models import Model
from speechbrain.pretrained import EncoderClassifier

def get_random_otp(NUMS, k):
    otp_pass = 0
    for num in random.sample(NUMS, k=k):
        otp_pass = otp_pass * 10 + num
    
    return otp_pass


def write_code(code, path):
    with open(path, 'w') as file:
        file.write(str(code))
        file.close()


def read_code(path):
    with open(path, 'r') as file:
        code = file.read()
        file.close()

    return code


def load_model_tensor(path):
    model = models.load_model(path)
    return model


def load_model_pytorch(ckpt_path):
    model = Model(lr=0.001)
    loaded_model = model.load_from_checkpoint(ckpt_path)
    return loaded_model


def load_decoder(path):
    decoder = load(open(path, 'rb'))
    return decoder


def resampler(waveform, orig_freq, new_freq=16000):
    transform  = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    waveform = transform(waveform)
    return waveform


def compute_embedding(classifier, fname):
    waveform, fs = torchaudio.load(fname)
    waveform = resampler(waveform, fs)
    embeddings = classifier.encode_batch(waveform)
    return embeddings


def extract_feature(path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="tmp/xvect"
    )
    feature_vector = compute_embedding(classifier, path)
    return feature_vector


def predict_speaker(model, decoder, feature_vector):
    vector = np.squeeze(feature_vector.numpy(), axis=1)
    pred_ids = model.predict(vector)
    speaker_idx = np.argmax(pred_ids)
    speaker = decoder.inverse_transform([speaker_idx])[0]
    return speaker


def load_and_transform(audio_path):
    want_sr = 16000
    melspect_func = T.MelSpectrogram(want_sr, n_fft=150)
    wave, sr = torchaudio.load(audio_path)
    wave = F.resample(wave, sr, want_sr)

    mel_spect = melspect_func(wave).squeeze()
    mel_spect = mel_spect.permute(1, 0)
    return mel_spect