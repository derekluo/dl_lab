# GPT-2 From Scratch

A from-scratch implementation of the GPT-2 architecture for language modeling and text generation.

## Project Structure

.
├── bin/
│   ├── concatenate.sh    # Data concatenation utility
│   └── download.py       # Data download script
├── data/                 # Training data directory
├── train.py             # Training script
├── inference.py         # Inference script
└── show_params.py       # Parameter visualization tool

## Dataset

The project uses a science fiction text dataset (`data/scifi.txt`) for training the language model. The dataset:

- Contains curated science fiction literature and stories
- Is preprocessed and tokenized for training
- Follows a specific format for model consumption
- Is automatically downloaded using the provided scripts

1. https://huggingface.co/datasets/wzy816/scifi
2. https://huggingface.co/datasets/zxbsmk/webnovel_cn


### Data Format

The text data is structured as:
- Plain text format (.txt)
- UTF-8 encoded
- One story/document per line
- Cleaned and normalized text

### Data Preparation

The data preparation pipeline:
1. Downloads raw text files using `bin/download.py`
2. Concatenates multiple sources using `bin/concatenate.sh`
3. Performs preprocessing and cleaning
4. Creates training-ready dataset

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLM-Lab.git
cd LLM-Lab
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download the training data:
```bash
python bin/download.py
```

2. Concatenate data files:
```bash
bash bin/concatenate.sh
```

### Training

To train the model:

```bash
python train.py
```

### Inference

To run inference with a trained model:

```bash
python inference.py
```

### Model Analysis

To visualize model parameters:

```bash
python show_params.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.