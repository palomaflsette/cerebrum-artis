"""
SAT Model Loader SIMPLIFICADO - Carrega modelo Show, Attend and Tell
"""
import torch
import pickle
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Adiciona path do SAT
artemis_sat_path = Path(__file__).parent.parent.parent.parent / 'garbage' / 'old_artemis-v2' / 'neural_speaker' / 'sat'
sys.path.insert(0, str(artemis_sat_path))

from artemis.neural_models.show_attend_tell import describe_model
from artemis.neural_models.resnet_encoder import ResnetEncoder
from artemis.neural_models.attentive_decoder import AttentiveDecoder, sample_captions


class SimpleArgs:
    """Args object para construir modelo SAT."""
    def __init__(self):
        self.vis_encoder = 'resnet34'
        self.atn_spatial_img_size = 7
        self.word_embedding_dim = 128
        self.rnn_hidden_dim = 512
        self.attention_dim = 512
        self.dropout_rate = 0.1
        self.teacher_forcing_ratio = 1.0
        self.use_emo_grounding = True
        self.emo_grounding_dims = [9, 9]  # input 9 emotions, output 9-dim (identity mapping)


class SATModelLoader:
    """
    Carrega modelo SAT (ResNet + LSTM com Attention).
    """
    
    def __init__(self, checkpoint_path, vocab_path, use_emotion_labels=True, device='cpu'):
        self.device = device
        
        # 1. Carrega vocab
        print(f"📚 Carregando vocabulário...")
        try:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"  ✅ Vocabulário: {len(self.vocab)} tokens")
        except ModuleNotFoundError as e:
            print(f"  ⚠️  Erro ao carregar vocab pickle (falta módulo artemis): {e}")
            print(f"  🔧 Usando fallback: extraindo vocab do checkpoint...")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extrai tamanho do vocab do decoder embedding
            decoder_emb_weight = checkpoint['model']['decoder.word_embedding.weight']
            vocab_size = decoder_emb_weight.shape[0]
            
            # Cria vocab simplificado
            class SimpleVocab:
                def __init__(self, size):
                    self.size = size
                    self.pad = 0
                    self.sos = 1
                    self.eos = 2
                    self.unk = 3
                    self.idx2word = {i: f'<token_{i}>' for i in range(size)}
                    self.idx2word[0] = '<pad>'
                    self.idx2word[1] = '<sos>'
                    self.idx2word[2] = '<eos>'
                    self.idx2word[3] = '<unk>'
                
                def __len__(self):
                    return self.size
            
            self.vocab = SimpleVocab(vocab_size)
            print(f"  ✅ Vocabulário simplificado: {vocab_size} tokens")
        
        # 2. Constrói modelo
        print("🧠 Construindo modelo SAT...")
        args = SimpleArgs()
        args.use_emo_grounding = use_emotion_labels
        
        # describe_model retorna nn.ModuleDict com 'encoder' e 'decoder'
        self.model = describe_model(self.vocab, args)
        
        # 3. Carrega checkpoint
        print(f"⚙️  Carregando checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        
        self.model.to(device)
        self.model.eval()
        
        print(f"  ✅ SAT carregado! (epoch {checkpoint.get('epoch', '?')})")
    
    def generate_caption(self, image_array, max_len=20, temperature=1.0, emotion=None):
        """
        Gera caption para imagem.
        
        Args:
            image_array: numpy array RGB (H, W, 3)
            max_len: comprimento máximo
            temperature: sampling temperature
            emotion: nome da emoção (opcional)
        
        Returns:
            String com a caption
        """
        # Prepara imagem
        img = Image.fromarray(image_array).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Cria um mini dataset com 1 imagem
        class SingleImageDataset:
            def __init__(self, img_tensor, emotion_onehot=None):
                self.img_tensor = img_tensor
                self.emotion = emotion_onehot
                self.batch_size = 1
            
            def __iter__(self):
                batch = {'image': self.img_tensor}
                if self.emotion is not None:
                    batch['emotion'] = self.emotion
                yield batch
                
            def __len__(self):
                return 1
        
        # Prepara emotion grounding (se usado)
        emotion_onehot = None
        if self.model['decoder'].uses_aux_data:
            emotion_map = {
                'amusement': 0, 'awe': 1, 'contentment': 2, 'excitement': 3,
                'anger': 4, 'disgust': 5, 'fear': 6, 'sadness': 7, 'something_else': 8
            }
            
            emo_idx = emotion_map.get(emotion, 8) if emotion else 8
            emotion_onehot = torch.zeros(1, 9, device=self.device)
            emotion_onehot[0, emo_idx] = 1.0
        
        # Cria dataset temporário
        dataset = SingleImageDataset(img_tensor, emotion_onehot)
        
        # Gera caption usando sample_captions
        with torch.no_grad():
            predictions, _ = sample_captions(
                self.model,
                dataset,
                max_utterance_len=max_len,
                sampling_rule='argmax',
                device=self.device,
                temperature=temperature,
                drop_unk=True,
                drop_bigrams=False
            )
        
        # Decodifica tokens para texto
        caption_tokens = predictions[0].cpu().numpy()
        
        words = []
        for token_idx in caption_tokens:
            if token_idx == self.vocab.eos:
                break
            if token_idx in [self.vocab.pad, self.vocab.sos]:
                continue
            
            # Tenta obter palavra do vocabulário
            word = self.vocab.idx2word.get(token_idx, f'<UNK_{token_idx}>')
            words.append(word)
        
        caption = ' '.join(words)
        return caption if caption else "an emotional artwork"
