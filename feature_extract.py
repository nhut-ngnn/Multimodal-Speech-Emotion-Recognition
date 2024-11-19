import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import torchaudio
import pickle
from transformers import BertTokenizer, BertModel
from speechbrain.pretrained import EncoderClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IEMOCAP dataset paths
IEMOCAP_TRAIN_PATH = "metadata/IEMOCAP_metadata_train.csv"
IEMOCAP_VAL_PATH = "metadata/IEMOCAP_metadata_val.csv"
IEMOCAP_TEST_PATH = "metadata/IEMOCAP_metadata_test.csv"

audio_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("All loaded")