import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

# 1. ENCODER
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Bidirectional = True
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        
        # Linear layer để gộp 2 chiều (Forward + Backward) thành 1 vector cho Decoder
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        
        # Pack sequence
        packed_embedded = pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=True)
        
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        
        # outputs: [src len, batch size, enc hid dim * 2]
        outputs, _ = pad_packed_sequence(packed_outputs)
        
        # Xử lý Hidden state ban đầu cho Decoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
      
        return outputs, hidden

# 2. ATTENTION 
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # Attention tính điểm dựa trên Hidden state của Decoder và Outputs của Encoder
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, dec hid dim]
        # encoder_outputs: [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) 
        encoder_outputs = encoder_outputs.permute(1, 0, 2) 
        # Tính Energy: E = tanh(Wx + Uy)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        # Tính Attention score
        attention = self.v(energy).squeeze(2) 
        
        return F.softmax(attention, dim=1)

# 3. DECODER 
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
    
        input = input.unsqueeze(0) 
        embedded = self.dropout(self.embedding(input))
        
        # Tính attention weights [batch, src len]
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1) # [batch, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Tính Weighted Context Vector: Tổng trọng số các từ nguồn
        weighted = torch.bmm(a, encoder_outputs) 
        weighted = weighted.permute(1, 0, 2)     
        
        # Đưa vào LSTM: [Embedding; Context]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), torch.zeros_like(hidden.unsqueeze(0))))
        
        # Dự đoán từ tiếp theo
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden.squeeze(0)

# 4. SEQ2SEQ
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
        
        # Encoder trả về outputs (cho Attention) và hidden (cho khởi tạo Decoder)
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        input = trg[0, :] # <sos>
        
        for t in range(1, trg_len):
            # Decoder nhận thêm encoder_outputs
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
            
        return outputs