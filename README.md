# RAVE Audio Generation Script

## Requirements

Create and activate a venv

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Audio from Prior Distribution

```bash
python generate.py --model path/to/model.ts --mode prior --duration 5 --temperature 0.8 --output_file /path/to/output.wav
```

### 2. Encode and Decode an Audio File

```bash
python generate.py --model path/to/model.ts --mode encode --input_file path/to/input.wav --output_file /path/to/output.wav
```

## Notes

- The script assumes a target sample rate of 48 kHz.
- If the input file is not mono, it will be converted to mono.
- Resampling is handled automatically if needed.
