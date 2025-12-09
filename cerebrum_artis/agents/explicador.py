"""
Agente Explicador - Explicabilidade Multimodal

Responsável pela explicabilidade (XAI) do sistema usando:
1. Explicação Textual - geração de texto explicando a emoção detectada
2. Explicação Visual (Grad-CAM) - mapas de calor mostrando regiões importantes
3. Análise de Features Fuzzy - contribuição de cada feature

Este agente explica PORQUÊ o modelo detectou determinada emoção.
"""

from typing import Dict, Union, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image


class Explicador:
    """
    Agente Explicador - XAI Multimodal
    
    Combina:
    - Explicação textual baseada em features
    - Grad-CAM para explicação visual
    - Análise de contribuição de features fuzzy
    """

    def __init__(self, language: str = 'pt'):
        """
        Inicializa o Explicador.

        Args:
            language: Idioma das explicações ('pt' ou 'en')
        """
        self.language = language
        self.emotion_templates = self._load_templates()
        self.fuzzy_feature_names = [
            'warmth', 'coldness', 'saturation', 'mutedness',
            'brightness', 'darkness', 'harmony'
        ]

    def _load_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Carrega templates de explicação para cada emoção.
        
        Returns:
            Dicionário com templates por emoção e idioma
        """
        templates = {
            'pt': {
                'amusement': [
                    "A obra evoca diversão através de {features}.",
                    "Elementos visuais como {features} criam uma atmosfera divertida.",
                    "A combinação de {features} transmite leveza e humor."
                ],
                'awe': [
                    "A imagem inspira admiração devido a {features}.",
                    "A grandiosidade é transmitida por {features}.",
                    "Elementos como {features} criam um senso de maravilhamento."
                ],
                'contentment': [
                    "A obra transmite contentamento através de {features}.",
                    "A harmonia visual criada por {features} evoca tranquilidade.",
                    "Elementos como {features} contribuem para uma sensação de paz."
                ],
                'excitement': [
                    "A energia da obra vem de {features}.",
                    "Elementos dinâmicos como {features} criam excitação.",
                    "A combinação de {features} transmite vibração e intensidade."
                ],
                'anger': [
                    "A obra expressa raiva através de {features}.",
                    "Elementos agressivos como {features} evocam tensão.",
                    "A intensidade de {features} transmite conflito e agitação."
                ],
                'disgust': [
                    "A obra evoca repulsa devido a {features}.",
                    "Elementos desconfortáveis como {features} criam aversão.",
                    "A combinação de {features} transmite desconforto visual."
                ],
                'fear': [
                    "A obra inspira medo através de {features}.",
                    "Elementos ameaçadores como {features} criam tensão.",
                    "A atmosfera sombria vem de {features}."
                ],
                'sadness': [
                    "A obra transmite tristeza através de {features}.",
                    "Elementos melancólicos como {features} evocam pesar.",
                    "A combinação de {features} cria uma atmosfera sombria."
                ],
                'something_else': [
                    "A obra evoca uma emoção complexa através de {features}.",
                    "Elementos visuais diversos como {features} criam uma resposta emocional única.",
                    "A combinação de {features} transmite uma emoção difícil de categorizar."
                ]
            },
            'en': {
                'amusement': [
                    "The artwork evokes amusement through {features}.",
                    "Visual elements like {features} create a playful atmosphere.",
                    "The combination of {features} conveys lightness and humor."
                ],
                'awe': [
                    "The image inspires awe due to {features}.",
                    "Grandeur is conveyed through {features}.",
                    "Elements like {features} create a sense of wonder."
                ],
                'contentment': [
                    "The artwork conveys contentment through {features}.",
                    "Visual harmony created by {features} evokes tranquility.",
                    "Elements like {features} contribute to a sense of peace."
                ],
                'excitement': [
                    "The artwork's energy comes from {features}.",
                    "Dynamic elements like {features} create excitement.",
                    "The combination of {features} conveys vibration and intensity."
                ],
                'anger': [
                    "The artwork expresses anger through {features}.",
                    "Aggressive elements like {features} evoke tension.",
                    "The intensity of {features} conveys conflict and agitation."
                ],
                'disgust': [
                    "The artwork evokes disgust due to {features}.",
                    "Uncomfortable elements like {features} create aversion.",
                    "The combination of {features} conveys visual discomfort."
                ],
                'fear': [
                    "The artwork inspires fear through {features}.",
                    "Threatening elements like {features} create tension.",
                    "The somber atmosphere comes from {features}."
                ],
                'sadness': [
                    "The artwork conveys sadness through {features}.",
                    "Melancholic elements like {features} evoke sorrow.",
                    "The combination of {features} creates a somber atmosphere."
                ],
                'something_else': [
                    "The artwork evokes a complex emotion through {features}.",
                    "Diverse visual elements like {features} create a unique emotional response.",
                    "The combination of {features} conveys an emotion difficult to categorize."
                ]
            }
        }
        return templates

    def explain_textual(
        self,
        emotion: str,
        confidence: float,
        fuzzy_features: Dict[str, float],
        top_n: int = 3
    ) -> str:
        """
        Gera explicação textual da predição.

        Args:
            emotion: Emoção predita
            confidence: Confiança da predição (0-1)
            fuzzy_features: Dicionário com valores das features fuzzy
            top_n: Número de features mais importantes para mencionar

        Returns:
            Texto explicativo em linguagem natural
        """
        # Ordena features por valor (contribuição)
        sorted_features = sorted(
            fuzzy_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Mapeia nomes das features para descrições
        feature_descriptions = self._get_feature_descriptions(
            sorted_features,
            self.language
        )

        # Seleciona template baseado na emoção
        emotion_key = emotion.lower().replace(' ', '_')
        templates = self.emotion_templates[self.language].get(
            emotion_key,
            self.emotion_templates[self.language]['something_else']
        )
        
        # Escolhe template baseado na confiança
        template_idx = min(int(confidence * len(templates)), len(templates) - 1)
        template = templates[template_idx]

        # Formata features
        features_text = self._format_features_list(feature_descriptions)

        # Gera explicação
        explanation = template.format(features=features_text)

        # Adiciona informação de confiança
        if self.language == 'pt':
            confidence_text = f" (confiança: {confidence:.1%})"
        else:
            confidence_text = f" (confidence: {confidence:.1%})"

        return explanation + confidence_text

    def _get_feature_descriptions(
        self,
        features: List[Tuple[str, float]],
        language: str
    ) -> List[str]:
        """
        Converte nomes técnicos de features para descrições em linguagem natural.
        """
        descriptions_map = {
            'pt': {
                'warmth': 'tons quentes',
                'coldness': 'tons frios',
                'saturation': 'cores saturadas',
                'mutedness': 'cores esmaecidas',
                'brightness': 'alta luminosidade',
                'darkness': 'tons escuros',
                'harmony': 'harmonia cromática'
            },
            'en': {
                'warmth': 'warm tones',
                'coldness': 'cold tones',
                'saturation': 'saturated colors',
                'mutedness': 'muted colors',
                'brightness': 'high brightness',
                'darkness': 'dark tones',
                'harmony': 'chromatic harmony'
            }
        }

        descriptions = []
        for feature_name, value in features:
            desc = descriptions_map[language].get(
                feature_name,
                feature_name
            )
            descriptions.append(f"{desc} ({value:.2f})")

        return descriptions

    def _format_features_list(self, features: List[str]) -> str:
        """
        Formata lista de features para texto natural.
        
        Exemplo: ['A', 'B', 'C'] -> 'A, B e C'
        """
        if len(features) == 0:
            return ""
        elif len(features) == 1:
            return features[0]
        elif len(features) == 2:
            connector = " e " if self.language == 'pt' else " and "
            return features[0] + connector + features[1]
        else:
            connector = " e " if self.language == 'pt' else " and "
            return ", ".join(features[:-1]) + connector + features[-1]

    def explain_visual_gradcam(
        self,
        model: nn.Module,
        image: torch.Tensor,
        target_layer: nn.Module,
        emotion_idx: int,
        **kwargs
    ) -> np.ndarray:
        """
        Gera mapa de calor Grad-CAM explicando a predição.

        Args:
            model: Modelo PyTorch
            image: Tensor da imagem (C, H, W)
            target_layer: Camada convolucional alvo para Grad-CAM
            emotion_idx: Índice da emoção predita
            **kwargs: Argumentos adicionais para o modelo (ex: input_ids, attention_mask)

        Returns:
            Mapa de calor Grad-CAM (H, W) normalizado entre 0-1
        """
        model.eval()

        # Hooks para capturar gradientes e ativações
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            # grad_output pode ser None em algumas camadas
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach())

        def forward_hook(module, input, output):
            activations.append(output.detach())

        # Registra hooks
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)

        # Forward pass
        image_batch = image.unsqueeze(0)  # Adiciona batch dimension
        image_batch.requires_grad = True  # Garante que gradientes sejam calculados
        
        # Prepara kwargs (adiciona batch dimension se necessário)
        model_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Se já tem batch dim, mantém. Se não, adiciona.
                if v.dim() == 1:
                    model_kwargs[k] = v.unsqueeze(0)
                else:
                    model_kwargs[k] = v
            else:
                model_kwargs[k] = v

        try:
            # Tenta passar kwargs (para modelos multimodais)
            output = model(image_batch, **model_kwargs)
        except TypeError:
            # Fallback para modelos que só aceitam imagem
            output = model(image_batch)

        # Se o output for uma tupla (comum em nossos modelos multimodais), pega o primeiro elemento (logits)
        if isinstance(output, tuple):
            output = output[0]

        # Backward pass para a classe alvo
        model.zero_grad()
        target = output[0, emotion_idx]
        target.backward(retain_graph=True)

        # Remove hooks
        backward_handle.remove()
        forward_handle.remove()

        # Verifica se capturamos gradientes
        if len(gradients) == 0 or len(activations) == 0:
            raise RuntimeError(
                f"Grad-CAM hooks não capturaram dados. "
                f"Gradients: {len(gradients)}, Activations: {len(activations)}. "
                f"Verifique se target_layer está correto."
            )

        # Calcula Grad-CAM
        grads = gradients[0].cpu().data.numpy()[0]  # (C, H, W)
        acts = activations[0].cpu().data.numpy()[0]  # (C, H, W)

        # Pesos: média dos gradientes em cada canal
        weights = np.mean(grads, axis=(1, 2))  # (C,)

        # Combinação linear ponderada
        cam = np.zeros(acts.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        # ReLU (apenas valores positivos)
        cam = np.maximum(cam, 0)

        # Normaliza para 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def visualize_gradcam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Sobrepõe mapa de calor Grad-CAM na imagem original.

        Args:
            image: Imagem original (H, W, 3) em RGB
            cam: Mapa de calor Grad-CAM (H, W)
            alpha: Transparência do mapa de calor (0-1)

        Returns:
            Imagem com overlay do mapa de calor (H, W, 3)
        """
        # Redimensiona CAM para tamanho da imagem
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Converte CAM para heatmap colorido
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Sobrepõe heatmap na imagem
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return overlay

    def explain_complete(
        self,
        model: nn.Module,
        image: Union[np.ndarray, torch.Tensor],
        emotion: str,
        emotion_idx: int,
        confidence: float,
        fuzzy_features: Dict[str, float],
        target_layer: Optional[nn.Module] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Gera explicação completa (textual + visual).

        Args:
            model: Modelo PyTorch
            image: Imagem (array numpy ou tensor)
            emotion: Nome da emoção predita
            emotion_idx: Índice da emoção
            confidence: Confiança da predição
            fuzzy_features: Features fuzzy extraídas
            target_layer: Camada para Grad-CAM (se None, usa última conv)
            save_path: Caminho para salvar visualização
            **kwargs: Argumentos adicionais para o modelo (ex: input_ids, attention_mask)

        Returns:
            Dicionário com explicação textual, CAM e visualização
        """
        # Explicação textual
        text_explanation = self.explain_textual(
            emotion,
            confidence,
            fuzzy_features
        )

        # Explicação visual (Grad-CAM)
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = image

        # Encontra última camada convolucional se não especificada
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)

        cam = self.explain_visual_gradcam(
            model,
            image_tensor,
            target_layer,
            emotion_idx,
            **kwargs
        )

        # Cria visualização
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image

        visualization = self.visualize_gradcam(image_np, cam)

        # Salva se requisitado
        if save_path:
            Image.fromarray(visualization).save(save_path)

        return {
            'text_explanation': text_explanation,
            'gradcam': cam,
            'visualization': visualization,
            'fuzzy_features': fuzzy_features,
            'emotion': emotion,
            'confidence': confidence
        }

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """
        Encontra a última camada convolucional do modelo.
        """
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv


# Classe mantida para compatibilidade
class ExplicadorVisual(Explicador):
    """Alias para Explicador (compatibilidade)"""
    
    """emotion: Emoção para explicar
        save_path: Se fornecido, salva mapa de calor

    Returns:
        Array numpy com mapa de calor
    """
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Sobrepõe mapa de calor na imagem original.

        Args:
            image: Imagem original
            heatmap: Mapa de calor Grad-CAM
            alpha: Transparência do heatmap [0, 1]

        Returns:
            Imagem com heatmap sobreposto
        """
        raise NotImplementedError()