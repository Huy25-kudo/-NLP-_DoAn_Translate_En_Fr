# data_utils.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os
import sys
import spacy
# 1. TẢI MODEL NGÔN NGỮ
def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"🔸 Đang tải gói ngôn ngữ {model_name}...")
        os.system(f"{sys.executable} -m spacy download {model_name}")
        return spacy.load(model_name)

spacy_en = load_spacy_model('en_core_web_sm')
spacy_fr = load_spacy_model('fr_core_news_sm')

# 3. TOKENIZER
def en_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def fr_tokenizer(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# 4. CLASS VOCABULARY
class Vocabulary:
    def __init__(self, freq_threshold=2, max_size=10000):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = 4
        
        # Đếm tần suất tất cả các từ
        for sentence in sentence_list:
            for word in tokenizer(sentence):
                frequencies[word] += 1
        
        # Lấy danh sách các từ phổ biến nhất theo max_size trừ đi 4 token đặc biệt
        common_words = frequencies.most_common(self.max_size - 4)
        
        for word, freq in common_words:
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def __call__(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def lookup_token(self, index):
        return self.itos.get(index, "<unk>")

def build_vocab(filepath, tokenizer):
    print(f"   - Đang đọc file {os.path.basename(filepath)} để xây từ điển ...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
    vocab = Vocabulary(freq_threshold=2, max_size=10000)
    sentences = [line.strip() for line in open(filepath, encoding='utf-8')]
    vocab.build_vocabulary(sentences, tokenizer)
    return vocab

# 5. DATASET & COLLATE
class Multi30kDataset(Dataset):
    def __init__(self, src_file, trg_file):
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"KHÔNG TÌM THẤY FILE: {src_file}")
        self.src_data = [line.strip() for line in open(src_file, 'r', encoding='utf-8')]
        self.trg_data = [line.strip() for line in open(trg_file, 'r', encoding='utf-8')]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        return self.src_data[index], self.trg_data[index]

class Collate:
    def __init__(self, src_vocab, trg_vocab, device):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device
        self.pad_idx = src_vocab.stoi["<pad>"]
        self.sos_idx = src_vocab.stoi["<sos>"]
        self.eos_idx = src_vocab.stoi["<eos>"]

    def text_transform(self, text, tokenizer, vocab):
        tokens = tokenizer(text)
        indices = vocab(tokens)
        return torch.tensor([self.sos_idx] + indices + [self.eos_idx], dtype=torch.long)

    def __call__(self, batch):
        src_batch, trg_batch = [], []
        for src_text, trg_text in batch:
            src_batch.append(self.text_transform(src_text, en_tokenizer, self.src_vocab))
            trg_batch.append(self.text_transform(trg_text, fr_tokenizer, self.trg_vocab))
      
        batch_zipped = sorted(zip(src_batch, trg_batch), key=lambda x: len(x[0]), reverse=True)
        src_batch, trg_batch = zip(*batch_zipped)

        src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
        src_padded = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=False)
        trg_padded = pad_sequence(trg_batch, padding_value=self.pad_idx, batch_first=False)

        return src_padded, trg_padded, src_lens