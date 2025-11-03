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
    def __init__(self, vocab_size, pad_idx, freeze_backbone=True):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim=512, freeze_backbone=freeze_backbone)
        self.decoder = DecoderRNN(vocab_size, embed_dim=512, hidden_dim=512, pad_idx=pad_idx)

    def forward(self, images, captions):
        feats = self.encoder(images)
        logits = self.decoder(feats, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=40, sos_id=1, eos_id=2, beam_size=1,
                 length_penalty_alpha=0.6, no_repeat_ngram_size=0, min_len=0,
                 banned_token_ids=None,
                 sampling=False, temperature=1.0, top_k=0, top_p=1.0):
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
            return self._generate_greedy(images, max_len, sos_id, eos_id,
                                         no_repeat_ngram_size, min_len, banned_token_ids,
                                         sampling, temperature, top_k, top_p)
        else:
            return self._generate_beam_search(images, max_len, sos_id, eos_id, beam_size,
                                             length_penalty_alpha, no_repeat_ngram_size, min_len, banned_token_ids)

    def _generate_greedy(self, images, max_len, sos_id, eos_id,
                          no_repeat_ngram_size=0, min_len=0, banned_token_ids=None,
                          sampling=False, temperature=1.0, top_k=0, top_p=1.0):
        device = images.device
        feats = self.encoder(images)
        feat_step = feats.unsqueeze(1)
        _, hidden = self.decoder.lstm(feat_step)

        B = images.size(0)
        cur = torch.full((B, 1), int(sos_id), dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            emb = self.decoder.embed(cur)
            out, hidden = self.decoder.lstm(emb, hidden)
            logits = self.decoder.fc(out.squeeze(1))
            # mask EOS before min_len
            if t < int(min_len):
                logits[:, int(eos_id)] = -1e9
            # mask banned tokens (e.g., UNK)
            if banned_token_ids:
                idx = torch.tensor(list(banned_token_ids), device=device, dtype=torch.long)
                logits.index_fill_(1, idx, -1e9)
            # no-repeat n-gram (simple bigram/trigram ban)
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                V = logits.size(-1)
                for i in range(B):
                    seq = sequences[i]
                    if len(seq) >= no_repeat_ngram_size - 1:
                        prefix = tuple(seq[-(no_repeat_ngram_size-1):])
                        # collect banned next tokens that would form a repeated n-gram
                        banned = set()
                        for s in range(len(seq) - (no_repeat_ngram_size - 1)):
                            if tuple(seq[s:s+no_repeat_ngram_size-1]) == prefix:
                                banned.add(seq[s+no_repeat_ngram_size-1])
                        if banned:
                            idx = torch.tensor(list(banned), device=device, dtype=torch.long)
                            logits[i, idx] = -1e9
            # Sampling / Argmax
            if sampling:
                # Apply temperature
                if temperature and temperature > 0 and temperature != 1.0:
                    logits = logits / float(temperature)

                logits_filtered = logits.clone()

                # Top-k filtering on logits (keep k largest, set rest to -inf)
                if top_k and top_k > 0:
                    k = min(int(top_k), logits_filtered.size(-1))
                    topk_vals, topk_idx = torch.topk(logits_filtered, k=k, dim=-1)
                    min_topk = topk_vals[..., -1].unsqueeze(-1)
                    keep = logits_filtered >= min_topk
                    logits_filtered = torch.where(keep, logits_filtered, torch.full_like(logits_filtered, float('-inf')))

                # Softmax to probabilities after top-k
                probs = torch.softmax(logits_filtered, dim=-1)

                # Top-p (nucleus) filtering on probs
                if top_p and top_p < 1.0:
                    tp = float(top_p)
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumsum <= tp
                    # garanta ao menos 1 token por linha
                    keep[..., 0] = True
                    filtered_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                    denom = filtered_sorted.sum(dim=-1, keepdim=True)
                    # se denom for 0 (numérico), volte para probs sem top-p
                    need_fallback = (denom <= 0)
                    if need_fallback.any():
                        filtered_sorted = sorted_probs
                        denom = filtered_sorted.sum(dim=-1, keepdim=True)
                    filtered_sorted = filtered_sorted / (denom + 1e-12)
                    probs = torch.zeros_like(probs).scatter(1, sorted_idx, filtered_sorted)

                # Última linha de defesa: se por algum motivo a soma for 0, volte ao softmax simples dos logits
                row_sums = probs.sum(dim=-1, keepdim=True)
                invalid = (row_sums <= 0) | ~torch.isfinite(row_sums)
                if invalid.any():
                    probs = torch.softmax(logits, dim=-1)

                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
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

    def _generate_beam_search(self, images, max_len, sos_id, eos_id, beam_size,
                               length_penalty_alpha=0.6, no_repeat_ngram_size=0, min_len=0, banned_token_ids=None):
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
            # mask banned tokens (e.g., UNK)
            if banned_token_ids:
                idx = torch.tensor(list(banned_token_ids), device=device, dtype=torch.long)
                logits.index_fill_(1, idx, -1e9)
            log_probs = torch.log_softmax(logits, dim=-1) # (B*k, V)

            if t == 0:
                log_probs = log_probs.view(B, beam_size, -1)[:, 0, :] # (B, V)
                # block EOS before min_len
                if t < int(min_len):
                    log_probs[:, int(eos_id)] = -1e9
                top_k_scores, top_k_tokens = log_probs.topk(beam_size, dim=1) # (B, k), (B, k)
                
                cur = top_k_tokens.view(-1, 1) # (B*k, 1)
                for i, token_idx in enumerate(top_k_tokens.flatten()):
                    sequences[i].append(token_idx.item())

            else:

                prev_scores = top_k_scores.view(-1, 1)  # (B*k,1)
                # Assemble candidate scores tensor (B, k, V)
                cand_scores = (prev_scores + log_probs).view(B, beam_size, -1)

                # no-repeat-ngram: mask banned tokens per beam
                if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                    V = cand_scores.size(-1)
                    for i in range(B):
                        for j in range(beam_size):
                            seq = sequences[i * beam_size + j]
                            if len(seq) >= no_repeat_ngram_size - 1:
                                prefix = tuple(seq[-(no_repeat_ngram_size-1):])
                                banned = set()
                                for s in range(len(seq) - (no_repeat_ngram_size - 1)):
                                    if tuple(seq[s:s+no_repeat_ngram_size-1]) == prefix:
                                        banned.add(seq[s+no_repeat_ngram_size-1])
                                if banned:
                                    idx = torch.tensor(list(banned), device=device, dtype=torch.long)
                                    cand_scores[i, j, idx] = -1e9

                # block EOS before min_len
                if t < int(min_len):
                    cand_scores[:, :, int(eos_id)] = -1e9

                # length penalty (GNMT): score / ((5+len)^{alpha} / 6^{alpha})
                if length_penalty_alpha and length_penalty_alpha > 0:
                    lens = torch.tensor([
                        len(sequences[i * beam_size + j]) + 1 for i in range(B) for j in range(beam_size)
                    ], device=device, dtype=torch.float32).view(B, beam_size, 1)
                    denom = ((5.0 + lens) ** length_penalty_alpha) / (6.0 ** length_penalty_alpha)
                    cand_scores = cand_scores / denom

                # Flatten and select top-k
                flat = cand_scores.view(B, -1)
                top_k_scores, top_k_indices = flat.topk(beam_size, dim=1) # (B,k)

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
