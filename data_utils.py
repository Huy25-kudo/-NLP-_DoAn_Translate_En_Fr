import torch
from torch.utils.data import Dataset
from collections import Counter
import os
import sys
import subprocess
import spacy


# --- 1. TẢI MODEL NGÔN NGỮ ---
try:
    spacy.load('en_core_web_sm')
    spacy.load('fr_core_news_sm')
except OSError:
    print("🔸 Đang tải gói ngôn ngữ Spacy...")
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download fr_core_news_sm")

def get_tokenizer(tokenizer_type, language='en_core_web_sm'):
    if tokenizer_type == 'spacy':
        spacy_model = spacy.load(language)
        # Trả về một hàm tokenizer
        return lambda text: [tok.text for tok in spacy_model.tokenizer(text)]
    else:
        raise ValueError("Code này chỉ hỗ trợ tokenizer='spacy'")

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

def tokenize_en(text):
    return en_tokenizer(text)

def tokenize_fr(text):
    return fr_tokenizer(text)

# --- 2. CLASS VOCABULARY ---
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in text
        ]

# --- 3. CLASS DATASET ---
class Multi30kDataset(Dataset):
    def __init__(self, src_file, trg_file, src_vocab=None, trg_vocab=None):
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"❌ KHÔNG TÌM THẤY FILE: {src_file}")
        
        print(f"🔹 Đang đọc: {os.path.basename(src_file)}...")
        self.src_data = [line.strip() for line in open(src_file, 'r', encoding='utf-8')]
        self.trg_data = [line.strip() for line in open(trg_file, 'r', encoding='utf-8')]
        
        self.src_tokenized = [tokenize_en(text) for text in self.src_data]
        self.trg_tokenized = [tokenize_fr(text) for text in self.trg_data]

        if src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build_vocabulary(self.src_tokenized)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocabulary()
            self.trg_vocab.build_vocabulary(self.trg_tokenized)
        else:
            self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        src_text = self.src_tokenized[index]
        trg_text = self.trg_tokenized[index]

        src_num = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.numericalize(src_text) + [self.src_vocab.stoi["<eos>"]]
        trg_num = [self.trg_vocab.stoi["<sos>"]] + self.trg_vocab.numericalize(trg_text) + [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src_num), torch.tensor(trg_num)

# --- 4. CLASS COLLATE ---
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        src = [item[0] for item in batch]
        trg = [item[1] for item in batch]
        
        src_lens = torch.tensor([len(x) for x in src])
        
        src = torch.nn.utils.rnn.pad_sequence(src, batch_first=False, padding_value=self.pad_idx)
        trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=False, padding_value=self.pad_idx)
        
        return src, trg, src_lens