import torchaudio
import numpy as np
import streamlit as st

from pickle import load
from keras import models
from torchaudio.transforms import Resample
from speechbrain.pretrained import EncoderClassifier


def load_model(path):
    model = models.load_model(path)
    return model


def load_decoder(path):
    decoder = load(open(path, 'rb'))
    return decoder


def resampler(waveform, orig_freq, new_freq=16000):
    transform  = Resample(orig_freq=orig_freq, new_freq=new_freq)
    waveform = transform(waveform)
    return waveform


def compute_embedding(classifier, fname):
    waveform, fs = torchaudio.load(fname)
    waveform = resampler(waveform, fs)
    embeddings = classifier.encode_batch(waveform)
    return embeddings

def extract_feature(path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    feature_vector = compute_embedding(classifier, path)
    return feature_vector


def predict_speaker(model, decoder, feature_vector):
    vector = np.squeeze(feature_vector.numpy(), axis=1)
    pred_ids = model.predict(vector)
    speaker_idx = np.argmax(pred_ids)
    speaker = decoder.inverse_transform([speaker_idx])[0]
    return speaker
