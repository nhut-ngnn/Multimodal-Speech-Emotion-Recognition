import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import torchaudio
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
from speechbrain.pretrained import EncoderClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IEMOCAP dataset paths
IEMOCAP_TRAIN_PATH = "metadata/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "metadata/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "metadata/IEMOCAP_metadata_test.csv"

audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("All loaded")

def extract_features(metadata_path, output_path):
    data_list = pd.read_csv(metadata_path)
    data_pkl = []
    with torch.no_grad():
        for idx in range(len(data_list)):
            audio_file = data_list['audio_file'][idx]
            audio_signal, _ = torchaudio.load(audio_file, normalize=True)
            audio_outputs = audio_model.encode_batch(audio_signal)
            audio_embed = audio_outputs.mean(axis=0)[0]
            label = data_list['label'][idx]
            label = torch.tensor(label)
            data_pkl.append({
                'audio_embed': audio_embed,
                'label': label
            })

    with open(output_path, "wb") as output_file:
        pickle.dump(data_pkl, output_file)

# Create training set for IEMOCAP
extract_features(IEMOCAP_TRAIN_PATH, "features/IEMOCAP_BERT_ECAPA_train.pkl")
