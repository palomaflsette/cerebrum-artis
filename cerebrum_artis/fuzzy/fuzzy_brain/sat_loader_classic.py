"""
SAT Model Loader - Carrega modelo SAT clássico (LSTM-based) treinado
"""
import torch
import pickle
import sys
from pathlib import Path
from argparse import Namespace

# Adiciona path do artemis-v2/neural_speaker/sat
artemis_path = Path(__file__).parent.parent.parent / 'artemis-v2' / 'neural_speaker' / 'sat'
sys.path.insert(0, str(artemis_path))

from artemis.neural_models.show_attend_tell import describe_model


class SimpleVocab:
    """Vocabulário simplificado compatível com o checkpoint."""
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.pad = 0
        self.sos = 1
        self.eos = 2
        self.unk = 3
        
    def __len__(self):
        return len(self.idx2word)


class SATModelLoader:
    """
    Carrega e executa o modelo SAT clássico (Show, Attend & Tell) treinado no ArtEmis.
    """
    
    def __init__(self, checkpoint_path, vocab_path=None, device='cuda'):
        """
        Args:
            checkpoint_path: Caminho para best_model.pt
            vocab_path: Caminho para vocabulary.pkl (pode ser None se extração do checkpoint funcionar)
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        
        # Carrega vocabulário
        print(f"📚 Carregando vocabulário de {vocab_path}...")
        try:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            # Verifica vocab size do checkpoint para garantir compatibilidade
            checkpoint_temp = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_vocab_size = checkpoint_temp['model']['decoder.word_embedding.weight'].shape[0]
            
            if len(self.vocab) != checkpoint_vocab_size:
                print(f"  ⚠️  Vocab size mismatch: pickle={len(self.vocab)}, checkpoint={checkpoint_vocab_size}")
                print(f"  🔧 Usando vocab size do checkpoint: {checkpoint_vocab_size}")
                
                # Expande vocab para tamanho do checkpoint
                for i in range(len(self.vocab), checkpoint_vocab_size):
                    token_name = f'<extra_{i}>'
                    self.vocab.word2idx[token_name] = i
                    self.vocab.idx2word[i] = token_name
                
            print(f"  ✅ Vocabulário: {len(self.vocab)} tokens")
        except Exception as e:
            print(f"  ⚠️  Falha ao carregar vocab pickle (esperado): {e}")
            print(f"  🔧 Extraindo vocabulário do checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vocab = self._extract_vocab_from_checkpoint(checkpoint)
            print(f"  ✅ Vocabulário: {len(self.vocab)} tokens")
        
        # Carrega checkpoint
        print(f"⚙️  Carregando checkpoint de {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reconstrói argumentos do modelo a partir do checkpoint
        self.args = self._reconstruct_args_from_checkpoint(checkpoint)
        
        # Constrói modelo
        print(f"🧠 Construindo modelo SAT clássico (LSTM-based)...")
        self.model = describe_model(self.vocab, self.args)
        
        # Carrega pesos
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modelo SAT carregado com sucesso!")
        
    def _extract_vocab_from_checkpoint(self, checkpoint):
        """Extrai vocabulário simplificado a partir do checkpoint."""
        # Pega vocab_size do embedding layer
        decoder_embedding = checkpoint['model'].get('decoder.word_embedding.weight')
        if decoder_embedding is None:
            raise ValueError("Não foi possível extrair vocab_size do checkpoint!")
        
        vocab_size = decoder_embedding.shape[0]
        print(f"  📊 Vocab size detectado: {vocab_size}")
        
        # Cria vocabulário genérico com tokens especiais
        word2idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        # Adiciona tokens genéricos (não temos os strings, mas não precisamos para inferência)
        for i in range(4, vocab_size):
            word2idx[f'<token_{i}>'] = i
        
        idx2word = {v: k for k, v in word2idx.items()}
        
        vocab = SimpleVocab(word2idx, idx2word)
        print(f"  ✅ Vocabulário simplificado criado: {len(vocab)} tokens")
        return vocab
    
    def _reconstruct_args_from_checkpoint(self, checkpoint):
        """Reconstrói os argumentos do modelo a partir do checkpoint."""
        model_state = checkpoint['model']
        
        # Detecta dimensões a partir dos pesos
        word_emb_dim = model_state['decoder.word_embedding.weight'].shape[1]
        
        # RNN hidden dim: LSTM tem 4 gates, então weight_hh é (4*hidden, hidden)
        rnn_hidden_dim = model_state['decoder.decode_step.weight_hh'].shape[1]
        
        # Attention dim: full_att.weight shape é (1, att_dim)
        attention_dim = model_state['decoder.attention.full_att.weight'].shape[1]
        
        # Detecta se usa emotion grounding
        has_emo_grounding = 'decoder.auxiliary_net.0.weight' in model_state
        emo_in_dim = 9 if has_emo_grounding else 0
        emo_out_dim = model_state['decoder.auxiliary_net.0.weight'].shape[0] if has_emo_grounding else 0
        
        # Detecta encoder: encoder_dim a partir de decoder.init_h input
        encoder_dim = model_state['decoder.init_h.weight'].shape[1]
        
        # Determina qual ResNet baseado no encoder_dim
        # ResNet34: 512, ResNet50: 2048
        if encoder_dim == 512:
            vis_encoder = 'resnet34'
        elif encoder_dim == 2048:
            vis_encoder = 'resnet50'
        else:
            vis_encoder = 'resnet34'  # Padrão
        
        args = Namespace(
            word_embedding_dim=word_emb_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            attention_dim=attention_dim,
            vis_encoder=vis_encoder,
            atn_spatial_img_size=None,  # Não usado
            use_emo_grounding=has_emo_grounding,
            emo_grounding_dims=[emo_in_dim, emo_out_dim],
            dropout_rate=0.1,  # Padrão
            teacher_forcing_ratio=1.0  # Não usado em inferência
        )
        
        print(f"  📐 Dimensões detectadas:")
        print(f"     - Word embedding: {word_emb_dim}")
        print(f"     - RNN hidden: {rnn_hidden_dim}")
        print(f"     - Attention: {attention_dim}")
        print(f"     - Encoder: {vis_encoder} (dim={encoder_dim})")
        print(f"     - Emotion grounding: {emo_in_dim} → {emo_out_dim}")
        
        return args
    
    @torch.no_grad()
    def generate(self, image_features, emotion_onehot=None, max_len=54, beam_size=5):
        """
        Gera caption para a imagem usando beam search.
        
        Args:
            image_features: Tensor (1, 2048, 7, 7) - features do ResNet34
            emotion_onehot: Tensor (1, 9) - one-hot encoding da emoção (opcional)
            max_len: Comprimento máximo da caption
            beam_size: Tamanho do beam para beam search
            
        Returns:
            str: Caption gerada
        """
        self.model.eval()
        
        # Encoder: extrai features
        encoder = self.model['encoder']
        decoder = self.model['decoder']
        
        # Aplica encoder
        enc_image = encoder(image_features.to(self.device))
        # enc_image: (1, H, W, 512) para resnet34 - encoder permuta para (B, H, W, C)
        
        # Reshape para (1, num_pixels, enc_dim) para attention
        batch_size, h, w, enc_dim = enc_image.shape
        enc_image = enc_image.view(batch_size, h * w, enc_dim)  # (1, 49, 512)
        
        # Prepara emotion se fornecida
        auxiliary_input = None
        if self.args.use_emo_grounding:
            if emotion_onehot is not None:
                auxiliary_input = emotion_onehot.to(self.device)
            else:
                # Se modelo usa emo grounding mas não foi fornecida, usa zeros (neutro)
                auxiliary_input = torch.zeros(1, self.args.emo_grounding_dims[0]).to(self.device)
        
        # Beam search
        batch_size = 1
        k = beam_size
        
        # Inicializa sequências com <sos>
        k_prev_words = torch.LongTensor([[self.vocab.sos]] * k).to(self.device)  # (k, 1)
        seqs = k_prev_words  # (k, 1)
        
        # Scores das sequências (log probabilities)
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)
        
        # Sequências completas (finalizadas com <eos>)
        complete_seqs = []
        complete_seqs_scores = []
        
        step = 1
        
        # Expande encoder features para k beams
        enc_image_exp = enc_image.expand(k, *enc_image.shape[1:])  # (k, 49, 512)
        
        # Estados LSTM iniciais
        h, c = decoder.init_hidden_state(enc_image_exp)  # (k, rnn_hidden_dim)
        
        # Beam search loop
        while True:
            # Pega última palavra de cada sequência
            embeddings = decoder.word_embedding(k_prev_words).squeeze(1)  # (k, word_emb_dim)
            
            # Attention + LSTM step
            awe, alpha = decoder.attention(enc_image_exp, h)
            
            # Concatena embedding + attention context + auxiliary (emotion)
            if auxiliary_input is not None:
                aux_exp = auxiliary_input.expand(k, -1)  # (k, emo_dim)
                aux_proj = decoder.auxiliary_net(aux_exp)  # (k, emo_ground_dim)
                gate = torch.sigmoid(decoder.f_beta(h))  # (k, enc_dim)
                awe = gate * awe
                lstm_input = torch.cat([embeddings, awe, aux_proj], dim=1)
            else:
                gate = torch.sigmoid(decoder.f_beta(h))
                awe = gate * awe
                lstm_input = torch.cat([embeddings, awe], dim=1)
            
            # LSTM step
            h, c = decoder.decode_step(lstm_input, (h, c))
            
            # Predição de próxima palavra
            scores = decoder.next_word(decoder.dropout(h))  # (k, vocab_size)
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            
            # Adiciona scores acumulados
            scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)
            
            # Para o primeiro step, todos os k beams são idênticos
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (k,)
            else:
                # Seleciona top k de todos os k*vocab_size possíveis
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (k,)
            
            # Converte para índices (beam_idx, word_idx)
            prev_word_inds = (top_k_words / len(self.vocab)).long()  # Qual beam
            next_word_inds = top_k_words % len(self.vocab)  # Qual palavra
            
            # Adiciona novas palavras às sequências
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)
            
            # Verifica quais sequências não terminaram (<eos>)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) 
                             if next_word != self.vocab.eos]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
            # Adiciona sequências completas
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            k -= len(complete_inds)  # Reduz k
            
            # Se todas terminaram ou atingiu max_len
            if k == 0 or step >= max_len:
                break
            
            # Continua com sequências incompletas
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            enc_image_exp = enc_image_exp[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
            step += 1
        
        # Seleciona melhor sequência
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()
        
        # Converte índices para palavras
        caption_tokens = []
        for idx in seq:
            if idx == self.vocab.sos:
                continue
            elif idx == self.vocab.eos:
                break
            else:
                # Pega palavra do vocabulário (não filtra mais tokens genéricos)
                token = self.vocab.idx2word.get(idx, '<unk>')
                # Só pula tokens especiais (<pad>, <unk>, etc) mas aceita tudo do vocab
                if not token.startswith('<') or token.startswith('<extra_'):
                    caption_tokens.append(token)
                elif token.startswith('<extra_'):
                    # Se for token extra que adicionamos, tenta usar como está
                    # (pode ser palavra que estava faltando no vocab)
                    pass
        
        caption = ' '.join(caption_tokens)
        return caption
