import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import sys
import time
import random
import spacy
import nltk
from nltk.translate.bleu_score import corpus_bleu

# ==============================================================================
# PHẦN 1: DATA UTILS (Xử lý dữ liệu, Tokenizer, Vocabulary, Dataset)
# ==============================================================================

# --- Tải Model Ngôn Ngữ Spacy ---
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
        return lambda text: [tok.text for tok in spacy_model.tokenizer(text)]
    else:
        raise ValueError("Code này chỉ hỗ trợ tokenizer='spacy'")

# Khởi tạo tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

def tokenize_en(text):
    return en_tokenizer(text)

def tokenize_fr(text):
    return fr_tokenizer(text)

# --- Class Vocabulary ---
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

# --- Class Dataset ---
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

# --- Class Collate ---
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

# ==============================================================================
# PHẦN 2: MODEL (Encoder, Decoder, Seq2Seq)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        # Pack sequence để xử lý độ dài thay đổi
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src, src_len)
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
            
        return outputs

# ==============================================================================
# PHẦN 3: MAIN (Train, Eval, Inference Loop)
# ==============================================================================

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train(model, iterator, optimizer, criterion, clip, scaler, device):
    model.train()
    epoch_loss = 0
    for i, (src, trg, src_len) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        if i % 50 == 0:
            print(f"   Batch {i}/{len(iterator)} | Loss: {loss.item():.4f}")
    return epoch_loss / len(iterator)

# Hàm hỗ trợ tính BLEU (trả về list tokens)
def translate_sentence_internal(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = sentence
    tokens = ["<sos>"] + tokens + ["<eos>"]
    src_indexes = src_vocab.numericalize(tokens)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    
    trg_indexes = [trg_vocab.stoi["<sos>"]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.stoi["<eos>"]:
            break
            
    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:-1] # Bỏ sos và eos

def calculate_bleu(data, src_vocab, trg_vocab, model, device):
    targets = []
    outputs = []
    print("   Đang tính BLEU...")
    # Lấy mẫu tối đa 200 câu để test cho nhanh
    for i in range(min(len(data), 200)):
        src = data.src_tokenized[i]
        trg = data.trg_tokenized[i]
        pred = translate_sentence_internal(src, src_vocab, trg_vocab, model, device)
        targets.append([trg])
        outputs.append(pred)
    return corpus_bleu(targets, outputs)

if __name__ == "__main__":
    # --- 1. Thiết lập ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔹 Đang sử dụng thiết bị: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data', 'multi30k_en_fr')
    
    # Đường dẫn file dữ liệu
    train_src = os.path.join(data_dir, 'train.en')
    train_trg = os.path.join(data_dir, 'train.fr')
    val_src = os.path.join(data_dir, 'val.en') 
    val_trg = os.path.join(data_dir, 'val.fr')

    # --- 2. Khởi tạo Dữ liệu ---
    print("\n1. KHỞI TẠO DỮ LIỆU")
    train_dataset = Multi30kDataset(train_src, train_trg)
    
    # Nếu có file validation thì dùng, không thì dùng tạm train_dataset để test code
    if os.path.exists(val_src):
        val_dataset = Multi30kDataset(val_src, val_trg, src_vocab=train_dataset.src_vocab, trg_vocab=train_dataset.trg_vocab)
    else:
        val_dataset = train_dataset

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=Collate(train_dataset.src_vocab.stoi["<pad>"]),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=Collate(train_dataset.src_vocab.stoi["<pad>"]),
                             num_workers=0, pin_memory=True)

    # --- 3. Khởi tạo Model ---
    print(f"\n2. KHỞI TẠO MODEL")
    INPUT_DIM = len(train_dataset.src_vocab)
    OUTPUT_DIM = len(train_dataset.trg_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    EPOCHS = 16
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.trg_vocab.stoi["<pad>"])
    scaler = torch.amp.GradScaler('cuda')

    # --- 4. Huấn luyện ---
    print(f"\n3. BẮT ĐẦU HUẤN LUYỆN")
    best_valid_loss = float('inf')
    patience = 3
    no_improve_epoch = 0
    model_save_path = os.path.join(current_dir, 'best-model.pth')
    
    # Kiểm tra nếu đã có model thì có thể load (optional), ở đây ta train lại từ đầu hoặc train tiếp
    # Nếu muốn chỉ Inference thì comment vòng for loop này lại và load model.
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, 1, scaler, device)
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for src, trg, src_len in val_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, src_len, trg, 0) # 0 = turn off teacher forcing
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                valid_loss += loss.item()
        valid_loss /= len(val_loader)
        
        mins, secs = divmod(time.time() - start_time, 60)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Đã lưu model mới (Loss: {valid_loss:.3f})")
        else:
            no_improve_epoch += 1
            print(f"⚠️ Loss không giảm ({no_improve_epoch}/{patience})")
        
        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        
        if no_improve_epoch >= patience:
            print("🛑 DỪNG SỚM (Early Stopping)!")
            break

    # --- 5. Đánh giá ---
    print("\n4. ĐÁNH GIÁ KẾT QUẢ")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Đã load lại best model để đánh giá.")
        
    score = calculate_bleu(val_dataset, train_dataset.src_vocab, train_dataset.trg_vocab, model, device)
    print(f'⭐️ BLEU Score = {score*100:.2f}')

    # --- 6. Inference Function (ĐÚNG YÊU CẦU ẢNH) ---
    def translate(sentence: str) -> str:
        """
        Hàm dịch với các yêu cầu:
        - Input: str
        - Output: str
        - Greedy Decoding
        - Max Length: 50
        - Stop at <eos>
        """
        # 1. Tokenize & Xử lý Input
        if isinstance(sentence, str):
            tokens = tokenize_en(sentence)
        else:
            tokens = sentence 

        # Thêm <sos>, <eos>
        tokens = ["<sos>"] + tokens + ["<eos>"]
        
        # Chuyển thành số
        src_indexes = train_dataset.src_vocab.numericalize(tokens)
        
        # Chuyển thành tensor
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
        src_len = torch.LongTensor([len(src_indexes)])

        # 2. Encoder
        model.eval()
        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor, src_len)

        # 3. Decoder Loop
        trg_indexes = [train_dataset.trg_vocab.stoi["<sos>"]]
        max_len = 50 
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            # Greedy decoding: Chọn token xác suất cao nhất
            pred_token = output.argmax(1).item()
            
            # Điều kiện dừng: Gặp <eos>
            if pred_token == train_dataset.trg_vocab.stoi["<eos>"]:
                break
            
            trg_indexes.append(pred_token)

        # 4. Detokenize (Số -> Chữ)
        trg_tokens = [train_dataset.trg_vocab.itos[i] for i in trg_indexes]
        
        # Loại bỏ <sos> ở đầu. (Lưu ý: <eos> đã break, không được add vào list nên không cần xóa đuôi)
        result_tokens = trg_tokens[1:]
        
        return " ".join(result_tokens)

    # --- 7. Chế độ tương tác ---
    print("\n-------------------------------------------")
    print("🤖 CHẾ ĐỘ DỊCH THỬ (Gõ 'q' để thoát)")
    print("-------------------------------------------")
    while True:
        try:
            sentence = input("\n🇬🇧 English: ")
            if sentence.lower() in ['q', 'quit', 'exit']: 
                break
            
            # Gọi hàm translate đúng chuẩn
            result = translate(sentence)
            
            print(f"🇫🇷 French:  {result}")
        except Exception as e: 
            print(f"❌ Lỗi: {e}")