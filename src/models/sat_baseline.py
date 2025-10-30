import torch, torch.nn as nn
import torchvision.models as models
from vocab import PAD, SOS, EOS

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=512, freeze_backbone=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(m.children())[:-1]   # remove fc => (B,2048,1,1)
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)     # (B,2048)
        feat = self.fc(feat)                   # (B,embed)
        return feat

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=1, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # features: (B,embed)  captions: (B,T)
        emb = self.embed(captions)             # (B,T,E)
        # prepend feature as first step (como 'context vector')
        feat = features.unsqueeze(1)           # (B,1,E)
        inp = torch.cat([feat, emb[:, :-1, :]], dim=1)  # teacher forcing
        out, _ = self.lstm(inp)
        logits = self.fc(out)                  # (B,T,V)
        return logits

class SATBaseline(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim=512, freeze_backbone=True)
        self.decoder = DecoderRNN(vocab_size, embed_dim=512, hidden_dim=512, pad_idx=pad_idx)

    def forward(self, images, captions):
        feats = self.encoder(images)
        logits = self.decoder(feats, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=40, sos_id=1, eos_id=2, beam_size=1):
        """
        Greedy/Beam search generation for captioning.

        Args:
            images: Tensor (B, C, H, W)
            max_len: maximum number of generated tokens
            sos_id: start-of-sequence token id
            eos_id: end-of-sequence token id
            beam_size: size of the beam for search. 1 means greedy search.

        Returns:
            List[List[int]] of length B with generated token ids (without SOS and EOS).
        """
        self.eval()
        if beam_size == 1:
            return self._generate_greedy(images, max_len, sos_id, eos_id)
        else:
            return self._generate_beam_search(images, max_len, sos_id, eos_id, beam_size)

    def _generate_greedy(self, images, max_len, sos_id, eos_id):
        device = images.device
        feats = self.encoder(images)
        feat_step = feats.unsqueeze(1)
        _, hidden = self.decoder.lstm(feat_step)

        B = images.size(0)
        cur = torch.full((B, 1), int(sos_id), dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.decoder.embed(cur)
            out, hidden = self.decoder.lstm(emb, hidden)
            logits = self.decoder.fc(out.squeeze(1))
            nxt = logits.argmax(dim=-1)

            for i in range(B):
                if not finished[i]:
                    token_i = nxt[i].item()
                    if token_i == int(eos_id):
                        finished[i] = True
                    else:
                        sequences[i].append(token_i)
            
            if finished.all():
                break
            cur = nxt.unsqueeze(1)
        return sequences

    def _generate_beam_search(self, images, max_len, sos_id, eos_id, beam_size):
        device = images.device
        B = images.size(0)
        
        feats = self.encoder(images) 
        
        feats = feats.repeat_interleave(beam_size, dim=0) # (B*k, E)
        feat_step = feats.unsqueeze(1) # (B*k, 1, E)
        _, hidden = self.decoder.lstm(feat_step) # (layers, B*k, H)

        cur = torch.full((B * beam_size, 1), int(sos_id), dtype=torch.long, device=device)

        top_k_scores = torch.zeros(B, beam_size, device=device)
        
        sequences = [[] for _ in range(B * beam_size)]
        
        for t in range(max_len):
            emb = self.decoder.embed(cur) # (B*k, 1, E)
            out, hidden = self.decoder.lstm(emb, hidden) # (B*k, 1, H)
            logits = self.decoder.fc(out.squeeze(1)) # (B*k, V)
            log_probs = torch.log_softmax(logits, dim=-1) # (B*k, V)

            if t == 0:
           
                log_probs = log_probs.view(B, beam_size, -1)[:, 0, :] # (B, V)
                top_k_scores, top_k_tokens = log_probs.topk(beam_size, dim=1) # (B, k), (B, k)
                
                cur = top_k_tokens.view(-1, 1) # (B*k, 1)
                for i, token_idx in enumerate(top_k_tokens.flatten()):
                    sequences[i].append(token_idx.item())

            else:

                prev_scores = top_k_scores.view(-1, 1)
                log_probs = prev_scores + log_probs # (B*k, V)

                log_probs = log_probs.view(B, -1) # (B, k*V)
                top_k_scores, top_k_indices = log_probs.topk(beam_size, dim=1) # (B, k)

                beam_indices = top_k_indices // logits.size(-1) # (B, k)
                token_indices = top_k_indices % logits.size(-1) # (B, k)

                new_sequences = [[] for _ in range(B * beam_size)]
                
                batch_offset = torch.arange(B, device=device) * beam_size
                reorder_indices = (beam_indices + batch_offset.unsqueeze(1)).flatten()

                hidden = (hidden[0][:, reorder_indices, :], hidden[1][:, reorder_indices, :])

                for i in range(B):
                    for j in range(beam_size):
                        beam_idx = beam_indices[i, j].item()
                        token_idx = token_indices[i, j].item()
                        
                        original_beam_global_idx = i * beam_size + beam_idx
                        new_beam_global_idx = i * beam_size + j
                        
                        new_sequences[new_beam_global_idx] = sequences[original_beam_global_idx] + [token_idx]

                sequences = new_sequences
                cur = token_indices.view(-1, 1) # (B*k, 1)

        final_sequences = []
        for i in range(B):
            best_seq = sequences[i * beam_size]
            if eos_id in best_seq:
                best_seq = best_seq[:best_seq.index(eos_id)]
            final_sequences.append(best_seq)
            
        return final_sequences
