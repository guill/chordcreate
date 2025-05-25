# ChordCreate

ChordCreate is a tool for intelligently generating chord assignments for words based on a given CharaChorder layout and word frequency data. It's designed to work with CharaChorder devices, creating optimal chord assignments based on word frequency, ergonomics, and other configurable parameters.

## Prerequisites

Before using ChordCreate, make sure you have:

1. Python 3.6+ installed
2. Required Python packages: `pip install networkx pydantic spacy tqdm pyyaml`
3. Download the SpaCy English model: `python -m spacy download en_core_web_md`
4. Download the [English Word Frequency dataset](https://www.kaggle.com/datasets/rtatman/english-word-frequency) from Kaggle and unzip it to `dataset/unigram_freq.csv`

## Setup

1. Export your CharaChorder layout from https://charachorder.io/config/settings/ and save it as `state/layout.json`
2. Generate a default configuration file:
   ```
   python chorder.py defaults > settings.yaml
   ```
3. (Optional) Edit the `settings.yaml` file to customize chord generation parameters

## Usage

ChordCreate offers several commands:

### Generate Chord Assignments

```
python chorder.py create [settings_file] --out chords.json
```

Parameters:
- `settings_file` (optional): Path to your YAML configuration file (defaults to built-in settings if not provided)
- `--out`: The output file to write the chord assignments to (default: `chords.json`)

### Print Default Configuration

```
python chorder.py defaults > settings.yaml
```

This will output the default configuration as YAML, which you can modify to customize the chord generation process.

### Convert Between Formats

```
python chorder.py convert input_file output_file --format FORMAT
```

Parameters:
- `input_file`: Path to the input chord file
- `output_file`: Path where the converted file will be saved
- `--format`: The format to convert to:
  - `raw`: Format used by CharaChorder's "Restore" button
  - `easy`: A more human-readable and easily parseable format
  - `need`: Format compatible with PowerToys Need plugin

## Output Formats

ChordCreate supports three output formats:

1. **Raw Format**: Compatible with CharaChorder's "Restore" button at charachorder.io/config/settings/
2. **Easy Format**: A more readable JSON format, easier to parse for custom applications
3. **Need Format**: Specifically formatted for use with the PowerToys Need plugin

## Configuration Options

The configuration file (`settings.yaml`) allows you to customize many aspects of chord generation:

- `top_n`: Number of top-frequency words to consider (default: 2000)
- `max_chord_size`: Maximum number of keys in a chord (default: 6)
- `min_chord_size`: Minimum number of keys in a chord (default: 2)
- `categories`: Word categories with customizable frequency multipliers
- `explicit_chords`: Manually specify chords for specific words
- `banned_chords`: Combinations that should never be assigned
- `heuristic`: Fine-tune the chord assignment algorithm

See the default configuration for a complete list of options and their descriptions.

## Example Workflow

1. Generate a default configuration:
   ```
   python chorder.py defaults > my_settings.yaml
   ```

2. Edit `my_settings.yaml` to customize parameters as needed

3. Generate chord assignments:
   ```
   python chorder.py create my_settings.yaml --out my_chords.json
   ```

4. Convert to raw format for uploading to CharaChorder:
   ```
   python chorder.py convert my_chords.json charachorder_upload.json --format raw
   ```

5. Upload to your CharaChorder device using the "Restore" button at charachorder.io/config/settings/

## Troubleshooting

- Ensure your `state/layout.json` file is valid and exported from the CharaChorder configuration page
- Check that the `dataset/unigram_freq.csv` file is properly formatted
- If you're getting errors about missing dependencies, run `pip install -r requirements.txt` (if provided) or install the required packages individually
