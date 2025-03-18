import torch
from math import ceil
import soundfile as sf
from librosa import resample
import numpy as np
import argparse
import random

fs = 48000

def sample_prior(model, n_steps, temperature):
    """
    This function returns a latent representation (an output tensor we get from a prior distribution)
    """
    # Temperature scaling
    inputs_rave = [torch.ones(1, 1, n_steps) * temperature]

    with torch.no_grad():
        prior = model.prior(inputs_rave[0])
    print(prior.shape)
    return prior

def get_audio_from_file(input_file):
    """
    Load and preprocess an audio file.
    """
    audio, sr = sf.read(input_file)
    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    # Resample if necessary
    if sr != fs:
        audio = resample(audio, orig_sr=sr, target_sr=fs)
    return audio

def encode_input_file(model, input_file):
    """
    This function encodes an input audio file into a latent representation.
    """
    audio = get_audio_from_file(input_file)
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    with torch.no_grad():
        latent = model.encode(audio_tensor)
    print(latent.shape)
    return latent

def decode(model, latent):
    """
    Decode a latent representation into audio.
    """
    with torch.no_grad():
        audio = model.decode(latent)
    return audio

def get_downsampling_ratio(model):
    x_len = 2**14
    n_channels = 1 # mono
    x = torch.zeros(1, n_channels, x_len)
    z = model.encode(x)
    downsampling_ratio = x_len // z.shape[-1]
    return downsampling_ratio

def time(dur, ratio):
    total_audio_samples = dur * fs
    steps = total_audio_samples / ratio
    return int(ceil(steps))

def write_audio_to_file(audio, output_file='output.wav'):
    audio_np = audio.squeeze().numpy()  # Remove batch and channel dimensions
    sf.write(output_file, audio_np, fs)

def apply_scale_and_bias(latent, scale, bias):
    for i in range(latent.size(1)):
        latent[:, i, :] = latent[:, i, :] * scale[i] + bias[i]
    return latent

def main():
    parser = argparse.ArgumentParser(description="Generate audio using a RAVE model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the TorchScript model file.")
    parser.add_argument('--mode', type=str, required=True, choices=['prior', 'encode'], 
                        help="Mode: 'prior' to generate from prior, 'encode' to encode/decode an audio file.")
    parser.add_argument('--duration', type=float, help="Duration of the generated audio (for prior mode).")
    parser.add_argument('--temperature', type=float, help="Temperature for sampling from the prior (for prior mode).")
    parser.add_argument('--input_file', type=str, help="Path to the input audio file (for encode mode).")
    parser.add_argument('--output_file', type=str, default='output.wav', help="Path to save the output audio file.")
    args = parser.parse_args()

    model = torch.jit.load(args.model)
    model.eval()

    if args.mode == 'prior' and not hasattr(model, 'prior'):
        print(f"The model {args.model} does not have a prior method.")
        exit()

    downsampling_ratio = get_downsampling_ratio(model)

    if args.mode == 'prior':
        if args.duration is None or args.temperature is None:
            print("Error: --duration and --temperature are required for prior mode.")
            exit()
        n_steps = time(args.duration, downsampling_ratio)
        latent = sample_prior(model, n_steps, args.temperature)

    elif args.mode == 'encode':
        if args.input_file is None:
            print("Error: --input_file is required for encode mode.")
            exit()
        latent = encode_input_file(model, args.input_file)

    latent_dim = latent.size(1)
    scale = [random.uniform(0, 2) for _ in range(latent_dim)]
    bias = [random.uniform(-10, 10) for _ in range(latent_dim)]
    latent = apply_scale_and_bias(latent, scale, bias)

    audio = decode(model, latent)
    write_audio_to_file(audio, args.output_file)
    print(f"Audio saved to {args.output_file}")

if __name__ == '__main__':
    main()
