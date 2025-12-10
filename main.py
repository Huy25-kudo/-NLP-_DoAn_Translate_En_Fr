import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import sys
from data_utils import Multi30kDataset, Collate, tokenize_en, install_package
from model import Encoder, Decoder, Seq2Seq
import nltk
from nltk.translate.bleu_score import corpus_bleu

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

# tính BLEU (trả về list token)
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
    return trg_tokens[1:-1]

def calculate_bleu(data, src_vocab, trg_vocab, model, device):
    targets = []
    outputs = []
    print("   Đang tính BLEU...")
    for i in range(min(len(data), 200)):
        src = data.src_tokenized[i]
        trg = data.trg_tokenized[i]
        pred = translate_sentence_internal(src, src_vocab, trg_vocab, model, device)
        targets.append([trg])
        outputs.append(pred)
    return corpus_bleu(targets, outputs)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔹 Đang sử dụng thiết bị: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data', 'multi30k_en_fr')
    train_src = os.path.join(data_dir, 'train.en')
    train_trg = os.path.join(data_dir, 'train.fr')
    val_src = os.path.join(data_dir, 'val.en') 
    val_trg = os.path.join(data_dir, 'val.fr')

    print("\n1. KHỞI TẠO DỮ LIỆU")
    train_dataset = Multi30kDataset(train_src, train_trg)
    val_dataset = Multi30kDataset(val_src, val_trg, src_vocab=train_dataset.src_vocab, trg_vocab=train_dataset.trg_vocab) if os.path.exists(val_src) else train_dataset

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=Collate(train_dataset.src_vocab.stoi["<pad>"]),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             collate_fn=Collate(train_dataset.src_vocab.stoi["<pad>"]),
                             num_workers=0, pin_memory=True)

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

    print(f"\n3. BẮT ĐẦU HUẤN LUYỆN")
    best_valid_loss = float('inf')
    patience = 3
    no_improve_epoch = 0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, 1, scaler, device)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for src, trg, src_len in val_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, src_len, trg, 0)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                valid_loss += loss.item()
        valid_loss /= len(val_loader)
        
        mins, secs = divmod(time.time() - start_time, 60)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best-model.pth'))
            print(f"Đã lưu model (Loss: {valid_loss:.3f})")
        else:
            no_improve_epoch += 1
            print(f" Loss không giảm ({no_improve_epoch}/{patience})")
        
        print(f'Epoch: {epoch+1:02} | Time: {int(mins)}m {int(secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        
        if no_improve_epoch >= patience:
            print("DỪNG SỚM (Early Stopping)!")
            break

    print("\n4. ĐÁNH GIÁ KẾT QUẢ")
    model.load_state_dict(torch.load(os.path.join(current_dir, 'best-model.pth')))
    score = calculate_bleu(val_dataset, train_dataset.src_vocab, train_dataset.trg_vocab, model, device)
    print(f' BLEU Score = {score*100:.2f}')

    def translate(sentence):
       
        # 1. Tokenize
        if isinstance(sentence, str):
            tokens = tokenize_en(sentence)
        else:
            tokens = sentence 

        # Thêm <sos>, <eos>
        tokens = ["<sos>"] + tokens + ["<eos>"]
        
        # Chuyển thành chỉ số 
        src_indexes = train_dataset.src_vocab.numericalize(tokens)
        
        # Chuyển thành tensor và thêm chiều batch
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
        src_len = torch.LongTensor([len(src_indexes)])

        # 2. Encoder
        model.eval()
        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor, src_len)

        # 3. Decoder
        trg_indexes = [train_dataset.trg_vocab.stoi["<sos>"]]
        max_len = 50 #
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            # Greedy decoding: Chọn token có xác suất cao nhất
            pred_token = output.argmax(1).item()
            
            # Điều kiện dừng: Gặp <eos>
            if pred_token == train_dataset.trg_vocab.stoi["<eos>"]:
                break
            
            trg_indexes.append(pred_token)

        # 4. Convert back to string
        trg_tokens = [train_dataset.trg_vocab.itos[i] for i in trg_indexes]
        
        # Bỏ <sos> ở đầu, ghép lại thành câu hoàn chỉnh
        # Hàm trả về string như yêu cầu
        return " ".join(trg_tokens[1:])

    print("\n CHẾ ĐỘ DỊCH THỬ (Gõ 'q' để thoát)")
    while True:
        try:
            sentence = input("\n English: ")
            if sentence.lower() in ['q', 'quit']: break
            
            # Gọi hàm translate mới vừa định nghĩa
            result = translate(sentence)
            
            print(f"🇫🇷 French:  {result}")
        except Exception as e: 
            print(f"Lỗi: {e}")