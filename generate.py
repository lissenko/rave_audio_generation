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
    return latent

def decode(model, latent):
    """
    Decode a latent representation into audio.
    """
    with torch.no_grad():
        audio = model.decode(latent)
    return audio

def get_model_ratio_and_dim(model):
    x_len = 2**14
    n_channels = 1 # mono
    x = torch.zeros(1, n_channels, x_len)
    z = model.encode(x)
    downsampling_ratio = x_len // z.shape[-1]
    return downsampling_ratio, z.size(1)

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

def get_rave_output(model, mode, duration, temperature, input_file, output_file, downsampling_ratio, scale, bias, noise_amount):

    model.eval()
    
    if mode == 'prior' and not hasattr(model, 'prior'):
        print(f"The model {model_path} does not have a prior method.")
        exit()

    if mode == 'prior':
        if duration is None or temperature is None:
            print("Error: --duration and --temperature are required for prior mode.")
            exit()
        n_steps = time(duration, downsampling_ratio)
        latent = sample_prior(model, n_steps, temperature)

    elif mode == 'encode':
        if input_file is None:
            print("Error: --input_file is required for encode mode.")
            exit()
        latent = encode_input_file(model, input_file)

    latent_dim = latent.size(1)
    latent = apply_scale_and_bias(latent, scale, bias)
    if noise_amount != 0.0:
        latent = latent + noise_amount * torch.randn_like(latent)

    audio = decode(model, latent)
    write_audio_to_file(audio, output_file)
    print(f"Audio saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate audio using a RAVE model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the TorchScript model file.")
    parser.add_argument('--mode', type=str, required=True, choices=['prior', 'encode'], 
                        help="Mode: 'prior' to generate from prior, 'encode' to encode/decode an audio file.")
    parser.add_argument('--duration', type=float, default=3.0, help="Duration of the generated audio (for prior mode).")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for sampling from the prior (for prior mode).")
    parser.add_argument('--noise', type=float, default=0.0, help="Noise to add to the latent representation.")
    parser.add_argument('--input_file', type=str, help="Path to the input audio file (for encode mode).")
    parser.add_argument('--output_file', type=str, default='output.wav', help="Path to save the output audio file.")
    parser.add_argument('--scale', type=float, nargs='+', default=[1.0], help="Scale factors for the latent space (default is [1.0]).")
    parser.add_argument('--bias', type=float, nargs='+', default=[0.0], help="Bias values for the latent space (default is [0.0]).")


    args = parser.parse_args()

    model = torch.jit.load(args.model)
    downsampling_ratio, latent_dim = get_model_ratio_and_dim(model)

    # Give the same value to every dim
    if len(args.scale) == 1:
        args.scale *= latent_dim
    if len(args.bias) == 1:
        args.bias *= latent_dim

    if len(args.scale) != latent_dim or len(args.bias) != latent_dim:
        print("Error: Scale and bias dimensions must match the latent dimension.")
        exit()

    get_rave_output(model, args.mode, args.duration, args.temperature, \
                    args.input_file, args.output_file, downsampling_ratio, \
                    args.scale, args.bias, args.noise)

    if False:

        # Check if latent_size exists in buffers
        latent_size_buffer = dict(model.named_buffers()).get("_rave.latent_size")
        if latent_size_buffer is not None:
            latent_size = latent_size_buffer.item()  # Convert tensor to integer
            print(f"Latent Size: {latent_size}")
        else:
            print("latent_size not found in buffers.")

        if hasattr(model, "full_latent_size"):
            print("FOUND: ", getattr(model, "full_latent_size"))

if __name__ == '__main__':
    main()
