from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import sampling_rate, audio_norm_target_dBFS
from encoder.params_data import mel_n_channels, mel_window_length, mel_window_step
from encoder.params_data import vad_window_length, vad_max_silence_length, vad_moving_average_width
from pathlib import Path
from typing import Optional, Union
import numpy as np
from utils import logmmse
import webrtcvad
import librosa
import struct
from scipy.io import wavfile

int16_max = (2 ** 15) - 1

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def trim_silence(wav, top_db=60):
    # top_db : set larger for clear speech, set small for noisy speech
    return librosa.effects.trim(wav, top_db=top_db, frame_length=512, hop_length=128)[0]

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=sampling_rate)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    # if source_sr is not None and source_sr != sampling_rate:
    #     wav = librosa.resample(wav, source_sr, sampling_rate)
    
    wav_abs_max = np.max(np.abs(wav))
    wav_abs_max = wav_abs_max if wav_abs_max > 0.0 else 1e-8
    wav = wav / wav_abs_max * 0.9
    # # Apply the preprocessing: normalize volume and shorten long silences 
    # wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    # wav = trim_long_silences(wav)
    # save_wav(wav, fpath_or_wav.name, sampling_rate) # TODO: rm DEBUG

    # denoise
    if len(wav) > sampling_rate*(0.3+0.1):
        noise_wav = np.concatenate([wav[:int(sampling_rate*0.15)],
                                    wav[-int(sampling_rate*0.15):]])
        profile = logmmse.profile_noise(noise_wav, sampling_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # trim silence
    wav = trim_silence(wav, 30) # top_db: smaller for noisy
    wav = trim_long_silences(wav)
    # save_wav(wav, fpath_or_wav.name.replace(".wav","_trimed.wav"), sampling_rate) # TODO: rm DEBUG
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == 1]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
