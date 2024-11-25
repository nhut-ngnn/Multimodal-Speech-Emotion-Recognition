import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import torch
import torchaudio
import ffmpeg
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm  # Import tqdm for progress bar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
MELD_TRAIN_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/Multimodal-Speech-Emotion-Recognition/metadata/MELD_metadata_convert.csv"
OUTPUT_DIR = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT_MODEL = BertModel.from_pretrained('bert-base-uncased').to(device)
AUDIO_MODEL = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def process_row(row, tokenizer, text_model, audio_model, device):
    text = row['raw_text']
    text_token = tokenizer(text, return_tensors="pt")
    text_token = text_token.to(device)
    text_outputs = text_model(**text_token)
    text_embeddings = text_outputs.last_hidden_state
    text_embed = text_embeddings[:, 0, :][0].cpu()

    audio_file = row['audio_file']
    audio_signal, _ = torchaudio.load(audio_file, normalize=True)
    audio_outputs = audio_model.encode_batch(audio_signal)
    audio_embed = audio_outputs.mean(axis=0)[0]

    label = row['label']
    label = torch.tensor(label)

    return {
        'text_embed': text_embed,
        'audio_embed': audio_embed,
        'label': label
    }

# Function to process a dataset and save as a pickle file
def process_dataset(input_path, output_path, tokenizer, text_model, audio_model, device):
    data_list = pd.read_csv(input_path)
    processed_data = []
    with torch.no_grad():
        # Add a progress bar here
        for idx, row in tqdm(data_list.iterrows(), total=len(data_list), desc="Processing dataset"):
            processed_data.append(process_row(row, tokenizer, text_model, audio_model, device))
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to {output_path}")

# Main function
def main():
    print("Loading models and tokenizer...")
    tokenizer = TOKENIZER
    text_model = TEXT_MODEL
    audio_model = AUDIO_MODEL

    print("Processing training set...")
    process_dataset(
        MELD_TRAIN_PATH,
        f"{OUTPUT_DIR}MELD_BERT_ECAPA_train.pkl",
        tokenizer,
        text_model,
        audio_model,
        device
    )

if __name__ == "__main__":
    main()
