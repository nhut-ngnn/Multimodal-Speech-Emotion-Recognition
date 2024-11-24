import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torchaudio
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TEXT_MAX_LENGTH = 100

# Paths
IEMOCAP_TRAIN_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/metadata/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/metadata/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/metadata/IEMOCAP_metadata_test.csv"

OUTPUT_PATH = "C:/Users/admin/Documents/Speech-Emotion_Recognition-2/features/"

# Load models
tokenizer_eng = BertTokenizer.from_pretrained('bert-base-uncased')
text_model_eng = BertModel.from_pretrained('bert-base-uncased').to(device)
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

print("Models loaded successfully")


def process_dataset(dataset_path, tokenizer, text_model, wav2vec_proc, wav2vec_mod, output_file):
    """Processes a dataset and saves the features as a pickle file."""
    dataset = pd.read_csv(dataset_path)
    processed_data = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Process text
            text = dataset['raw_text'][idx]
            text_token = tokenizer(text, return_tensors="pt", truncation=True, max_length=TEXT_MAX_LENGTH)
            text_token = text_token.to(device)
            text_outputs = text_model(**text_token)
            text_embed = text_outputs.last_hidden_state[:, 0, :][0].cpu()

            # Process audio
            audio_file = dataset['audio_file'][idx]
            audio_signal, _ = torchaudio.load(audio_file, normalize=True)
            audio_signal = audio_signal.to(device)
            
            inputs = wav2vec_proc(audio_signal.squeeze().cpu(), return_tensors="pt", sampling_rate=16000)
            audio_outputs = wav2vec_mod(**inputs)
            audio_embed = audio_outputs.last_hidden_state.mean(axis=1)[0].cpu()

            # Get label
            label = dataset['label'][idx]
            label = torch.tensor(label)

            processed_data.append({
                'text_embed': text_embed,
                'audio_embed': audio_embed,
                'label': label
            })

    # Save to pickle
    with open(output_file, "wb") as file:
        pickle.dump(processed_data, file)
    print(f"Processed data saved to {output_file}")


def main():
    """Main function to process IEMOCAP datasets."""
    # Process training set
    process_dataset(
        IEMOCAP_TRAIN_PATH,
        tokenizer_eng,
        text_model_eng,
        wav2vec_processor,
        wav2vec_model,
        f"{OUTPUT_PATH}IEMOCAP_BERT_wav2vec_train.pkl"
    )

    # Process validation set
    process_dataset(
        IEMOCAP_VAL_PATH,
        tokenizer_eng,
        text_model_eng,
        wav2vec_processor,
        wav2vec_model,
        f"{OUTPUT_PATH}IEMOCAP_BERT_wav2vec_val.pkl"
    )

    # Process test set
    process_dataset(
        IEMOCAP_TEST_PATH,
        tokenizer_eng,
        text_model_eng,
        wav2vec_processor,
        wav2vec_model,
        f"{OUTPUT_PATH}IEMOCAP_BERT_wav2vec_test.pkl"
    )


if __name__ == "__main__":
    main()
