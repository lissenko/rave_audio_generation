# RAVE Latent Space Explorer

![RAVE Latent Space Explorer](./demo/screenshot.png)

## Requirements

Create and activate a venv

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Audio from Prior Distribution

```bash
python generate.py \
  --model path/to/model.ts \
  --mode prior \
  --duration 5 \
  --temperature 0.8 \
  --noise 2.2 \
  --scale 1.0 1.2 1.5 \
  --bias 0.0 0.1 0.2 \
  --output_file /path/to/output.wav
```

### 2. Encode and Decode an Audio File

```bash
python generate.py \
  --model path/to/model.ts \
  --mode encode \
  --input_file path/to/input.wav \
  --noise 2.2 \
  --scale 1.0 1.2 1.5 \
  --bias 0.0 0.1 0.2 \
  --output_file /path/to/output.wav
```

## Parameters

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to the TorchScript model file.
  --mode {prior,encode}
                        Mode: 'prior' to generate from prior, 'encode' to encode/decode an audio file.
  --duration DURATION   Duration of the generated audio (for prior mode, default: 3.0).
  --temperature TEMPERATURE
                        Temperature for sampling from the prior (for prior mode, default: 1.0).
  --noise NOISE         Noise to add to the latent representation (default: 0.0).
  --input_file INPUT_FILE
                        Path to the input audio file (for encode mode).
  --output_file OUTPUT_FILE
                        Path to save the output audio file (default: output.wav).
  --scale SCALE [SCALE ...]
                        Scale factors for the latent space (default: [1.0] * latent_size).
  --bias BIAS [BIAS ...]
                        Bias values for the latent space (default: [0.0] * latent_size).


## User Interface

In addition to the command-line interface, the script can also be run
with a Flask-based web interface. This interface provides a user-friendly way
to interact with the model and tweak parameters without needing to use the
command line.

1. Run the Flask app

```bash
flask run
```

2. Access the User Interface: Open your browser and navigate to the URL where the server is running (http://127.0.0.1:5000/). 

## Notes

- The script assumes a target sample rate of 48 kHz.
- If the input file is not mono, it will be converted to mono.
- Resampling is handled automatically if needed.
- Ensure that the number of scale and bias values matches the latent dimension of the model. If not, the script will exit with an error.
