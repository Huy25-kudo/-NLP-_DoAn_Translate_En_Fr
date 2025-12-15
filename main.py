# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import random
import matplotlib.pyplot as plt 
import warnings
from nltk.translate.bleu_score import sentence_bleu
from model import Encoder, Decoder, Attention, Seq2Seq
warnings.filterwarnings("ignore")
from data_utils import Multi30kDataset, Collate, build_vocab, en_tokenizer, fr_tokenizer
from model import Encoder, Decoder, Seq2Seq
import torch
import torch.nn.functional as F

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, iterator, optimizer, criterion, clip, device, scaler):
    model.train()
    epoch_loss = 0
    use_amp = (device.type == 'cuda')

    for i, (src, trg, src_len) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
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
        else:
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_len in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, src_len, trg, 0) # Tắt teacher forcing khi eval
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# HÀM VẼ BIỂU ĐỒ
# def plot_history(train_history, valid_history, save_path='loss_chart.png'):
#     """
#     Vẽ biểu đồ Loss huấn luyện và kiểm thử
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_history, label='Training Loss', color='blue', linestyle='-')
#     plt.plot(valid_history, label='Validation Loss', color='orange', linestyle='--')
    
#     plt.title('Training & Validation Loss Curve')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     plt.savefig(save_path)
#     print(f" Đã lưu biểu đồ loss tại: {save_path}")
#     plt.close()

def evaluate_and_report(model, iterator, trg_vocab, device):
    print("\n-------------------------------------------")
    print(" ĐANG TẠO BÁO CÁO ĐÁNH GIÁ ...")
    model.eval()
    
    total_bleu = 0
    examples = []
    trg_itos = trg_vocab.itos
    
    with torch.no_grad():
        for src, trg, src_len in iterator:
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, src_len, trg, 0) 
            output = output.argmax(dim=2)        
            
            trg = trg.transpose(0, 1).cpu().numpy()
            output = output.transpose(0, 1).cpu().numpy()
            
            for i in range(trg.shape[0]):
                ref_tokens = []
                for idx in trg[i]:
                    if idx in [0, 1]: continue 
                    if idx == 2: break         
                    ref_tokens.append(trg_itos[idx])
                
                pred_tokens = []
                for idx in output[i]:
                    if idx == 2: break        
                    if idx not in [0, 1]:      
                        pred_tokens.append(trg_itos[idx])
                
                #  Tính BLEU 
                score = sentence_bleu([ref_tokens], pred_tokens)
                total_bleu += score
                
                examples.append({
                    'ref': " ".join(ref_tokens),
                    'pred': " ".join(pred_tokens),
                    'bleu': score
                })

    avg_bleu = total_bleu / len(examples)
    print(f" BLEU Score trung bình (Test): {avg_bleu:.4f}")
    return avg_bleu


# 1. HÀM DỊCH BEAM SEARCH
def beam_search_translate(sentence, model, src_vocab, trg_vocab, device, beam_width=3, max_len=50, alpha=1.0):
    model.eval()
    
    # Bước 1: Xử lý đầu vào 
    tokens = en_tokenizer(sentence.lower())
    if not tokens: return ""
    src_indices = src_vocab(tokens)
    # Thêm <sos> và <eos>
    src_tensor = torch.tensor([src_vocab.stoi["<sos>"]] + src_indices + [src_vocab.stoi["<eos>"]], dtype=torch.long).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_tensor)], dtype=torch.long)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    # Bước 2: Khởi tạo Beam
    beam = [(0, trg_vocab.stoi["<sos>"], hidden, [])]

    # Bước 3: Vòng lặp Decoding
    for _ in range(max_len):
        candidates = []
        
        # Duyệt qua các nhánh hiện tại trong beam
        for score, token_idx, hid, seq in beam:
            # Nếu nhánh đã gặp <eos>, giữ nguyên nhánh đó 
            if token_idx == trg_vocab.stoi["<eos>"]:
                candidates.append((score, token_idx, hid, seq))
                continue

            # Chuẩn bị input cho decoder
            input_tensor = torch.tensor([token_idx], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Decoder cần: input, hidden cũ, và encoder_outputs
                output, new_hidden = model.decoder(input_tensor, hid, encoder_outputs)
            
            # Tính Log Softmax
            probs = F.log_softmax(output, dim=1) 
            topk_probs, topk_ids = probs.topk(beam_width)

            # Mở rộng nhánh
            for k in range(beam_width):
                new_score = score + topk_probs[0][k].item()
                new_token = topk_ids[0][k].item()
                candidates.append((new_score, new_token, new_hidden, seq + [new_token]))

        # Bước 4: Sắp xếp và Cắt tỉa (Pruning)
        ordered = sorted(candidates, key=lambda x: x[0] / ((len(x[3]) + 1) ** alpha), reverse=True)
        
        # Lấy top k nhánh tốt nhất
        beam = ordered[:beam_width]

        # Điều kiện dừng sớm: Nếu tất cả các nhánh trong beam đều đã kết thúc bằng <eos>
        if all(x[1] == trg_vocab.stoi["<eos>"] for x in beam):
            break

    # --- Bước 5: Chọn kết quả tốt nhất ---
    # Tìm nhánh có điểm normalized cao nhất
    best_score = -float('inf')
    best_seq = None
    
    for score, token_idx, _, seq in beam:
        # Length Normalization: score / (len^alpha)
        norm_score = score / (len(seq) ** alpha)
        if norm_score > best_score:
            best_score = norm_score
            best_seq = seq
    # Chuyển index về từ vựng
    tokens = [trg_vocab.lookup_token(idx) for idx in best_seq if idx != trg_vocab.stoi["<eos>"]]
    return " ".join(tokens)


# 2. HÀM DỊCH GREEDY
def translate(sentence, model, src_vocab, trg_vocab, device):
    model.eval()
    try:
        tokens = en_tokenizer(sentence.lower()) 
        if not tokens: return ""
        src_indices = src_vocab(tokens)
    except:
        return "<error>"

    src_tensor = torch.tensor([src_vocab.stoi["<sos>"]] + src_indices + [src_vocab.stoi["<eos>"]], dtype=torch.long).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_tensor)], dtype=torch.long)

    with torch.no_grad():
        # Lấy encoder_outputs cho Attention
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    trg_indices = [trg_vocab.stoi["<sos>"]]
    
    for _ in range(50):
        trg_tensor = torch.tensor([trg_indices[-1]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            
        pred_token = output.argmax(1).item()
        
        if pred_token == trg_vocab.stoi["<eos>"]:
            break
            
        trg_indices.append(pred_token)

    return " ".join([trg_vocab.lookup_token(idx) for idx in trg_indices[1:]])

# CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔹 DEVICE: {device}")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    # Đường dẫn data
    data_dir = os.path.join(current_dir, 'data', 'multi30k_en_fr')
    train_src = os.path.join(data_dir, 'train.en')
    train_trg = os.path.join(data_dir, 'train.fr')
    val_src = os.path.join(data_dir, 'val.en') 
    val_trg = os.path.join(data_dir, 'val.fr')
    test_src = os.path.join(data_dir, 'test.en')
    test_trg = os.path.join(data_dir, 'test.fr')

    if not os.path.exists(train_src):
        print(f" Lỗi: Không tìm thấy dữ liệu tại {train_src}")
        sys.exit(1)

    print("\n1. ĐANG XỬ LÝ DỮ LIỆU...")
    src_vocab = build_vocab(train_src, en_tokenizer)
    trg_vocab = build_vocab(train_trg, fr_tokenizer)
    
    train_dataset = Multi30kDataset(train_src, train_trg)
    val_dataset = Multi30kDataset(val_src, val_trg)
    test_dataset = Multi30kDataset(test_src, test_trg) if os.path.exists(test_src) else val_dataset

    collate_fn = Collate(src_vocab, trg_vocab, device)
    
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\n2. KHỞI TẠO MÔ HÌNH (SEQ2SEQ + ATTENTION)...")
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    # Các tham số Dimension
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512 # Encoder hidden
    DEC_HID_DIM = 512 # Decoder hidden
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    
    # 2. Khởi tạo Encoder & Decoder
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    
    # 3. Kết hợp thành Seq2Seq
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Init weights 
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
                
    model.apply(init_weights)
    model.apply(init_weights)
    print(f"   - Trainable parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi["<pad>"])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    print("\n3. BẮT ĐẦU HUẤN LUYỆN...") 
    EPOCHS = 20
    save_path = os.path.join(current_dir, 'best_model.pth')
    chart_path = os.path.join(current_dir, 'training_chart.png')
    
    best_valid_loss = float('inf')
    early_stop_patience = 3
    early_stop_counter = 0
    train_history = []
    valid_history = []


    header = f"{'Epoch':^7} | {'Time':^9} | {'Train Loss':^12} | {'Val Loss':^12} | {'Status':<20}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for epoch in range(EPOCHS):
        start_time = time.time()
   
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 1, device, scaler)
        valid_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(valid_loss)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        
        mins, secs = divmod(time.time() - start_time, 60)
        time_str = f"{int(mins):02}m {int(secs):02}s"
        
        # Xử lý trạng thái lưu model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_counter = 0 
            torch.save(model.state_dict(), save_path)
            status = "Lưu Model"
        else:
            early_stop_counter += 1
            status = f"Wait ({early_stop_counter}/{early_stop_patience})"
        
        # --- IN DÒNG DỮ LIỆU ---
        print(f"{epoch+1:^7} | {time_str:^9} | {train_loss:^12.4f} | {valid_loss:^12.4f} | {status:<20}")
        
        if early_stop_counter >= early_stop_patience:
            print("-" * len(header))
            print(f"Early Stopping kích hoạt! Val loss không giảm sau {early_stop_patience} epochs.")
            break
            
    print("-" * len(header))

    
    # print("\n Đang vẽ biểu đồ huấn luyện...")
    # plot_history(train_history, valid_history, save_path=chart_path)

    print("\n4. KẾT QUẢ CUỐI CÙNG")
    model.load_state_dict(torch.load(save_path, map_location=device))
    evaluate_and_report(model, test_loader, trg_vocab, device)

    # DỊCH THỬ 
    print("\n-------------------------------------------")
    print("dịch thử (Gõ 'q' để thoát)")
    while True:
        try:
            text = input("\n English: ")
            if text.lower() in ['q', 'quit']: break
            trans = beam_search_translate(text, model, src_vocab, trg_vocab, device, beam_width=5) #Dùng beam search
            # trans = translate(text, model, src_vocab, trg_vocab, device) #Dùng greedy
            print(f"French:  {trans}")
        except Exception as e:
            print(f" Error: {e}")