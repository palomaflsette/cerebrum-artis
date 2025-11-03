import os, torch, random, pandas as pd, argparse
import cv2
import numpy as np
import torch
from packaging import version
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import PROJECT_SPLITS_DIR, PROJECT_CSV_DIR
from datasets.artemis import ArtemisDataset
from models.sat_baseline import SATBaseline as CaptionerSAT    
from vocab import load_vocab, PAD, SOS, EOS, UNK

def parse_args():
    p = argparse.ArgumentParser(description="Avaliar amostras e gerar Grad-CAM")
    p.add_argument("--ckpt", type=str, default="results/baseline_sat_best.pt", help="Caminho do checkpoint do modelo")
    p.add_argument("--n", type=int, default=6, help="Número de amostras aleatórias")
    p.add_argument("--max-len", type=int, default=40, help="Comprimento máximo da legenda gerada")
    p.add_argument("--beam-size", type=int, default=1, help="Tamanho do beam (1 = greedy)")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Proíbe repetição de n-gramas (0=desliga)")
    p.add_argument("--min-len", type=int, default=0, help="Comprimento mínimo antes de permitir EOS")
    p.add_argument("--sampling", action="store_true", help="Ativa amostragem em vez de argmax (quando beam=1)")
    p.add_argument("--top-k", type=int, default=0, help="Top-K para amostragem")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-P (nucleus) para amostragem")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperatura da amostragem")
    return p.parse_args()

# Wrapper para o modelo ser compatível com Grad-CAM
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, pad_idx, max_len):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.max_len = max_len

    def forward(self, images, captions=None):
        # O forward do Grad-CAM não precisa de captions, mas o nosso modelo sim.
        # Vamos criar um caption dummy para a chamada inicial.
        # Garantir que os gradientes fluam a partir da imagem (necessário para CAM quando o backbone está congelado)
        if not images.requires_grad:
            images.requires_grad_(True)
        if captions is None:
            captions = torch.full((images.size(0), self.max_len), self.pad_idx, device=images.device, dtype=torch.long)
        
        return self.model(images, captions)

# Alvo para o Grad-CAM focar na geração da legenda
class CaptioningTarget:
    def __init__(self, caption_ids):
        # Adiciona SOS ao início da legenda para o cálculo da perda
        self.caption_ids = torch.cat([
            torch.full((caption_ids.size(0), 1), SOS_ID, device=caption_ids.device, dtype=torch.long),
            caption_ids
        ], dim=1)

    def __call__(self, model_output):
        # model_output é (B, T, V)
        # Queremos calcular a perda para a legenda fornecida
        crit = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
        
        # O grad-cam pode 'espremer' a dimensão temporal quando aug_smooth=True
        # Se vier (T, V), reintroduzimos o batch como 1: (1, T, V)
        if model_output.ndim == 2:
            model_output = model_output.unsqueeze(0)

        # Comprimento alvo (inclui SOS no início)
        target_len_with_sos = int(self.caption_ids.size(1))  # = 1 + len(pred_ids)
        # Alvos reais para a perda: remove SOS
        target_captions_full = self.caption_ids[:, 1:].contiguous()  # (B, L)

        # Comprimento efetivo disponível em logits e alvos
        logits_time = int(model_output.size(1)) if model_output.dim() >= 3 else 0
        targets_time = int(target_captions_full.size(1))
        min_time = min(logits_time, targets_time)

        # Caso de borda: se algum lado for zero, não há gradiente útil a extrair
        if min_time <= 0:
            # Retorna um escalar conectado ao grafo para permitir backprop, porém com gradiente zero
            return model_output.sum() * 0.0

        # Trunque ambos para o mesmo comprimento
        output_logits_for_loss = model_output[:, :min_time, :].contiguous()   # (B, min_time, V)
        target_captions = target_captions_full[:, :min_time].contiguous()     # (B, min_time)

        loss = crit(
            output_logits_for_loss.view(-1, output_logits_for_loss.size(-1)),
            target_captions.view(-1)
        )
        return loss

def denormalize_image(tensor):
    # Desnormalização padrão da ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

def main():
    args = parse_args()
    global SOS_ID, PAD_ID
    voc  = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    itos = voc["itos"]
    PAD_ID = itos.index(PAD)
    SOS_ID = itos.index(SOS)
    eos_id = itos.index(EOS)
    unk_id = itos.index(UNK)

    ds = ArtemisDataset(split="val", image_size=224, max_len=args.max_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Modelo
    vocab_size = len(itos)
    model_base = CaptionerSAT(vocab_size=vocab_size, pad_idx=PAD_ID)

    # Carregar checkpoint com tolerância a mudanças de vocabulário.
    # Se o vocab atual for maior que o do checkpoint, copiamos o prefixo
    # das matrizes (assumindo que o vocabulário antigo é prefixo do novo
    # quando apenas reduzimos o min_freq). Demais entradas ficam randômicas.
    kwargs = {}
    if version.parse(torch.__version__) >= version.parse("2.4.0"):
        kwargs["weights_only"] = True

    ckpt = torch.load(args.ckpt, map_location="cpu", **kwargs)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    def _expand_decoder_if_needed(ckpt_sd, model):
        sd_new = model.state_dict()
        keys = [
            ("decoder.embed.weight", 2),
            ("decoder.fc.weight", 2),
            ("decoder.fc.bias", 1),
        ]
        changed = False
        for k, ndim in keys:
            if k in ckpt_sd and k in sd_new:
                w_old = ckpt_sd[k]
                w_new = sd_new[k]
                if w_old.shape != w_new.shape:
                    # Copia o prefixo (eixo vocab na dim 0)
                    rows = min(w_old.shape[0], w_new.shape[0])
                    if ndim == 2:
                        tmp = w_new.clone()
                        tmp[:rows, :w_old.shape[1]] = w_old[:rows]
                    else:
                        tmp = w_new.clone()
                        tmp[:rows] = w_old[:rows]
                    ckpt_sd[k] = tmp
                    changed = True
        return changed

    try:
        model_base.load_state_dict(ckpt, strict=True)
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" in msg or "Missing key(s)" in msg or "Unexpected key(s)" in msg:
            print("[warn] Incompatibilidade de shapes ao carregar checkpoint. Tentando ajustar camadas dependentes do vocabulário…")
            changed = _expand_decoder_if_needed(ckpt, model_base)
            # Carrega de forma não estrita para ignorar pequenas diferenças (ex: buffers)
            model_base.load_state_dict(ckpt, strict=False)
            if changed:
                print("[ok] Ajuste aplicado: pesos do decoder expandidos para o novo vocabulário.")
            else:
                print("[warn] Não foi possível ajustar automaticamente. Verifique se o vocab.json corresponde ao checkpoint.")
        else:
            raise

    model_base.to(device).eval()

    # --- Wrapper e Grad-CAM
    model = ModelWrapper(model_base, PAD_ID, args.max_len)
    # Use uma camada conv rica (layer4) em vez do avgpool para Grad-CAM
    target_layers = [model.model.encoder.backbone[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)

    os.makedirs("samples", exist_ok=True)
    rows = []

    idxs = random.sample(range(len(ds)), k=min(args.n, len(ds)))

    for j, i in enumerate(idxs):
        img_tensor, _, cap_ids = ds[i] 
        img_for_cam = img_tensor.unsqueeze(0).to(device)
        gt_tokens = [itos[int(t)] for t in cap_ids.tolist() if t not in (PAD_ID, SOS_ID, eos_id)]
        gt = " ".join(gt_tokens)

        # --- Geração com opções
        pred_ids = model.model.generate(
            img_for_cam,
            max_len=args.max_len,
            sos_id=SOS_ID,
            eos_id=eos_id,
            beam_size=args.beam_size,
            banned_token_ids=[unk_id],
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            min_len=args.min_len,
            sampling=args.sampling,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )[0]
        
        pred = " ".join(itos[t] if 0 <= t < len(itos) else UNK for t in pred_ids)

        # Métrica simples: taxa de UNK na predição
        unk_count = sum(1 for t in pred_ids if t == unk_id)
        pred_len = max(1, len(pred_ids))
        unk_rate = unk_count / pred_len

        # --- Geração do Grad-CAM (pule se a legenda ficou vazia)
        save_image(img_tensor, f"samples/img_{j}.png")
        if len(pred_ids) > 0:
            pred_ids_tensor = torch.tensor([pred_ids], device=device, dtype=torch.long)
            targets = [CaptioningTarget(pred_ids_tensor)]
            try:
                # Para permitir backward no LSTM com cuDNN, entre em modo train e desabilite cuDNN
                prev_training = model.model.training
                prev_cudnn = torch.backends.cudnn.enabled
                model.model.train()
                torch.backends.cudnn.enabled = False
                with torch.enable_grad():
                    grayscale_cam = cam(input_tensor=img_for_cam, targets=targets, aug_smooth=True, eigen_smooth=True)
                grayscale_cam = grayscale_cam[0, :] # Pegar o primeiro (e único) mapa
                # Restaurar estados
                torch.backends.cudnn.enabled = prev_cudnn
                if not prev_training:
                    model.model.eval()
                # Visualização
                rgb_img = denormalize_image(img_tensor)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cv2.imwrite(f"samples/img_{j}_cam.png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            except Exception as e:
                # Caso ocorra algum problema no CAM, apenas siga adiante
                print(f"[warn] Grad-CAM falhou para amostra {j}: {e}")
        
        rows.append((f"samples/img_{j}.png", pred, gt, f"{unk_rate:.3f}", pred_len))

    pd.DataFrame(rows, columns=["img","pred","gt","unk_rate_pred","len_pred"]).to_csv("samples/predicoes.csv", index=False)
    print(f"→ {args.n} amostras salvas em `samples/` (com e sem Grad-CAM) e predições em `samples/predicoes.csv` | beam={args.beam_size}")

if __name__ == "__main__":
    main()