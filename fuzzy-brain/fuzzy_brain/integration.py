"""
============================================================================
NEURAL-FUZZY INTEGRATION - FUSÃO DE MODELOS
============================================================================
Combina o modelo neural (SAT) com o sistema fuzzy para predição final.

ARQUITETURA HÍBRIDA:
--------------------
                    ┌──────────────┐
                    │   IMAGEM     │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────▼─────┐            ┌─────▼─────┐
        │ RESNET-50 │            │  VISUAL   │
        │  (CNN)    │            │ FEATURES  │
        └─────┬─────┘            └─────┬─────┘
              │                         │
        ┌─────▼─────┐            ┌─────▼─────┐
        │    SAT    │            │   FUZZY   │
        │  (Neural) │            │  SYSTEM   │
        └─────┬─────┘            └─────┬─────┘
              │                         │
              │  p_neural               │  p_fuzzy
              │  [9 emoções]            │  [9 emoções]
              │                         │
              └────────────┬────────────┘
                           │
                      ┌────▼────┐
                      │  FUSION │
                      └────┬────┘
                           │
                      p_final = fusion(p_neural, p_fuzzy)
                      
ESTRATÉGIAS DE FUSÃO:
---------------------
1. WEIGHTED SUM (v1.0 - IMPLEMENTADO):
   p_final = α·p_neural + (1-α)·p_fuzzy
   
   Vantagens:
   - Simples, rápido, interpretável
   - α ajustável (ex: 0.7 neural, 0.3 fuzzy)
   
   Desvantagens:
   - Não aproveita concordância/conflito
   - Peso fixo para todas as imagens

2. GUIDED FUZZY (v2.0 - FUTURO):
   Usa p_neural para modular hedges:
   
   SE p_neural[sadness] > 0.7:
       hedges = ["muito", "extremamente"]
   SE p_neural[sadness] < 0.3:
       hedges = []
   
   Vantagens:
   - Neural GUIA o fuzzy
   - Mantém interpretabilidade
   - Publicável!
   
3. CONFLICT ANALYSIS (v3.0 - PESQUISA):
   Detecta quando neural e fuzzy discordam:
   
   conflict = KL_divergence(p_neural || p_fuzzy)
   
   SE conflict > threshold:
       # Imagem ambígua, difícil
       p_final = ensemble(p_neural, p_fuzzy, method='vote')
   SENÃO:
       # Concordância, alta confiança
       p_final = p_neural (maior acurácia)

REFERÊNCIAS:
------------
- Nauck & Kruse (1999). Neuro-fuzzy systems for function approximation
- Jang (1993). ANFIS: adaptive-network-based fuzzy inference system
- Lin & Lee (1996). Neural Fuzzy Systems (Cap. 8 - Hybrid Learning)

============================================================================
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import warnings

# Import relativo ou absoluto
try:
    from .extractors.visual import VisualFeatureExtractor
    from .fuzzy.system import FuzzyInferenceSystem
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fuzzy_brain.extractors.visual import VisualFeatureExtractor
    from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem


class HybridEmotionPredictor:
    """
    Preditor híbrido Neural-Fuzzy para emoções em arte.
    
    Combina:
    - SAT (Show, Attend and Tell) - modelo neural treinado
    - Fuzzy System - regras interpretáveis baseadas em features visuais
    
    Attributes:
        visual_extractor: Extrator de features visuais
        fuzzy_system: Sistema de inferência fuzzy
        sat_model: Modelo SAT (carregado do checkpoint)
        device: 'cuda' ou 'cpu'
        fusion_weight: Peso para fusão (alpha)
        emotion_names: Lista de 9 emoções
    """
    
    def __init__(
        self,
        sat_checkpoint_path: Optional[str] = None,
        fusion_weight: float = 0.7,
        use_guided_fuzzy: bool = False,
        adaptive_fusion: bool = True,
        device: str = 'cpu'
    ):
        """
        Inicializa o preditor híbrido.
        
        Args:
            sat_checkpoint_path: Caminho para checkpoint do SAT.
                                Se None, usa apenas fuzzy (sem neural).
            fusion_weight: Peso PADRÃO do modelo neural na fusão (alpha).
                          p_final = alpha*p_neural + (1-alpha)*p_fuzzy
                          Default: 0.7 (70% neural, 30% fuzzy)
            use_guided_fuzzy: Se True, usa regras guiadas (v2.0).
            adaptive_fusion: Se True, ajusta peso baseado em características
                           da imagem (conteúdo semântico vs formal).
                           Default: True
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.fusion_weight = fusion_weight
        self.adaptive_fusion = adaptive_fusion
        
        # Lista de emoções (ArtEmis dataset)
        self.emotion_names = [
            'amusement', 'awe', 'contentment', 'excitement',
            'anger', 'disgust', 'fear', 'sadness', 'something_else'
        ]
        
        # Inicializa componentes
        print("⏳ Inicializando componentes híbridos...")
        
        # 1. Visual Feature Extractor (sempre necessário)
        print("  📊 Carregando Visual Feature Extractor...")
        self.visual_extractor = VisualFeatureExtractor()
        
        # 2. Fuzzy System
        print(f"  🧠 Inicializando Fuzzy System (guided={use_guided_fuzzy})...")
        self.fuzzy_system = FuzzyInferenceSystem(use_guided=use_guided_fuzzy)
        
        # 3. SAT Model (opcional)
        if sat_checkpoint_path is not None:
            print(f"  🤖 Carregando SAT model de {sat_checkpoint_path}...")
            self.sat_model = self._load_sat_model(sat_checkpoint_path)
        else:
            print("  ⚠️  SAT model não fornecido - usando apenas fuzzy!")
            self.sat_model = None
        
        print("✅ Sistema híbrido inicializado!\n")
    
    # ========================================================================
    # PREDIÇÃO PRINCIPAL
    # ========================================================================
    
    def predict(
        self, 
        image_path: str,
        return_components: bool = False
    ) -> Dict[str, float]:
        """
        Prediz emoções para uma imagem usando o sistema híbrido.
        
        Args:
            image_path: Caminho para imagem
            return_components: Se True, retorna também p_neural e p_fuzzy
        
        Returns:
            Se return_components=False:
                Dict com 9 emoções [0,1] somando 1.0
            
            Se return_components=True:
                Dict com:
                    'final': distribuição final
                    'neural': distribuição do SAT (ou None)
                    'fuzzy': distribuição do fuzzy
                    'fusion_weight': alpha usado (adaptativo ou fixo)
                    'features': features visuais extraídas
        
        Example:
            >>> predictor = HybridEmotionPredictor(
            ...     sat_checkpoint_path='sat_logs/sat_combined/checkpoints/best_model.pt',
            ...     fusion_weight=0.7
            ... )
            >>> emotions = predictor.predict('path/to/painting.jpg')
            >>> print(f"Sadness: {emotions['sadness']:.3f}")
        """
        # 1. Extrai features visuais
        features = self.visual_extractor.extract_all(image_path)
        
        # 2. Inferência fuzzy
        p_fuzzy = self.fuzzy_system.infer(features)
        
        # 3. Inferência neural (se disponível)
        if self.sat_model is not None:
            p_neural = self._predict_neural(image_path)
        else:
            p_neural = None
        
        # 4. Calcula peso de fusão (adaptativo ou fixo)
        if self.adaptive_fusion and p_neural is not None:
            alpha = self._compute_adaptive_weight(features)
        else:
            alpha = self.fusion_weight if p_neural is not None else 0.0
        
        # 5. Fusão
        p_final = self._fuse_predictions(p_neural, p_fuzzy, alpha)
        
        # Retorna
        if return_components:
            return {
                'final': p_final,
                'neural': p_neural,
                'fuzzy': p_fuzzy,
                'fusion_weight': alpha,
                'features': features
            }
        else:
            return p_final
    
    # ========================================================================
    # COMPONENTES
    # ========================================================================
    
    def _load_sat_model(self, checkpoint_path: str):
        """
        Carrega modelo SAT do checkpoint.
        
        TODO: Implementar carregamento real do SAT.
        Por enquanto, placeholder que retorna None.
        
        Args:
            checkpoint_path: Caminho para checkpoint
        
        Returns:
            Modelo SAT carregado
        """
        # TODO: Implementar carregamento real
        # Precisa do código do SAT (neural_speaker/sat/)
        
        warnings.warn(
            "Carregamento do SAT ainda não implementado! "
            "Retornando None (apenas fuzzy será usado)."
        )
        
        return None
    
    def _predict_neural(self, image_path: str) -> Optional[Dict[str, float]]:
        """
        Prediz emoções usando o modelo neural SAT.
        
        TODO: Implementar predição real.
        
        Args:
            image_path: Caminho para imagem
        
        Returns:
            Dict com 9 emoções [0,1] ou None se falhar
        """
        if self.sat_model is None:
            return None
        
        # TODO: Implementar predição real do SAT
        # 1. Preprocessar imagem
        # 2. Passar pelo ResNet-50
        # 3. Passar pelo SAT
        # 4. Retornar logits/probabilidades
        
        warnings.warn("Predição neural não implementada ainda!")
        return None
    
    def _fuse_predictions(
        self,
        p_neural: Optional[Dict[str, float]],
        p_fuzzy: Dict[str, float],
        alpha: float = None
    ) -> Dict[str, float]:
        """
        Combina predições neural e fuzzy.
        
        Implementa fusão simples (weighted sum):
            p_final = α·p_neural + (1-α)·p_fuzzy
        
        Args:
            p_neural: Probabilidades do modelo neural (ou None)
            p_fuzzy: Probabilidades do sistema fuzzy
            alpha: Peso do neural (se None, usa self.fusion_weight)
        
        Returns:
            Distribuição final [0,1] somando 1.0
        """
        # Se não tem neural, retorna apenas fuzzy
        if p_neural is None:
            return p_fuzzy
        
        # Define alpha
        if alpha is None:
            alpha = self.fusion_weight
        
        # Weighted sum
        p_final = {}
        
        for emotion in self.emotion_names:
            p_final[emotion] = (
                alpha * p_neural[emotion] + 
                (1 - alpha) * p_fuzzy[emotion]
            )
        
        # Normaliza (garantir soma = 1.0)
        total = sum(p_final.values())
        p_final = {e: p / total for e, p in p_final.items()}
        
        return p_final
    
    # ========================================================================
    # FUSÃO ADAPTATIVA
    # ========================================================================
    
    def _compute_adaptive_weight(self, features: Dict[str, float]) -> float:
        """
        Calcula peso de fusão adaptativamente baseado nas features.
        
        HEURÍSTICA:
        -----------
        Imagens com conteúdo SEMÂNTICO rico (complexas, coloridas,
        representacionais) → maior peso NEURAL (CNN vê conteúdo)
        
        Imagens FORMAIS/abstratas (simples, monocromáticas, geométricas)
        → maior peso FUZZY (regras capturam estética formal)
        
        INDICADORES de conteúdo SEMÂNTICO:
        - Alta complexidade (muitos elementos)
        - Alta saturação (cores vibrantes)
        - Baixa simetria (composição assimétrica = narrativa)
        - Alta rugosidade (textura detalhada)
        
        INDICADORES de composição FORMAL:
        - Baixa complexidade (minimalista)
        - Baixa saturação (monocromático)
        - Alta simetria (geométrico)
        - Baixa rugosidade (liso)
        
        Args:
            features: Features visuais extraídas
        
        Returns:
            alpha ∈ [0.5, 0.9]
                0.5 = abstrato puro (50% neural, 50% fuzzy)
                0.9 = representacional (90% neural, 10% fuzzy)
        
        Example:
            Franz Kline (abstrato P&B):
                complexity=0.03, saturation=0.05, symmetry=0.76
                → semantic_score = 0.2 → alpha = 0.58
            
            Otto Dix (guerra, figuras):
                complexity=0.25, saturation=0.35, symmetry=0.76
                → semantic_score = 0.5 → alpha = 0.70
        """
        # Extrai features relevantes
        complexity = features['complexity']
        saturation = features['saturation']
        symmetry = features['symmetry']
        roughness = features['texture_roughness']
        
        # Score semântico (0 = formal, 1 = semântico)
        # Quanto maior, mais provável que tenha conteúdo narrativo
        semantic_score = (
            0.3 * complexity +           # Complexidade visual
            0.3 * saturation +           # Cores (vs monocromático)
            0.2 * (1 - symmetry) +       # Assimetria (narrativa)
            0.2 * roughness              # Textura detalhada
        )
        
        # Mapeia score → alpha
        # semantic_score ∈ [0, 1] → alpha ∈ [0.5, 0.9]
        # Linear: alpha = 0.5 + 0.4 * semantic_score
        
        alpha_min = 0.5  # Mínimo (abstrato puro)
        alpha_max = 0.9  # Máximo (representacional)
        alpha = alpha_min + (alpha_max - alpha_min) * semantic_score
        
        # Clamp
        alpha = max(alpha_min, min(alpha_max, alpha))
        
        return alpha
    
    # ========================================================================
    # ANÁLISE E DEBUG
    # ========================================================================
    
    def explain_prediction(self, image_path: str) -> str:
        """
        Gera explicação detalhada da predição.
        
        Args:
            image_path: Caminho para imagem
        
        Returns:
            String com explicação completa
        """
        result = self.predict(image_path, return_components=True)
        
        explanation = []
        explanation.append("=" * 70)
        explanation.append("EXPLICAÇÃO DA PREDIÇÃO HÍBRIDA")
        explanation.append("=" * 70)
        explanation.append(f"\n📷 Imagem: {Path(image_path).name}\n")
        
        # Features visuais
        features = result['features']
        explanation.append("📊 FEATURES VISUAIS:")
        for feature, value in features.items():
            explanation.append(f"  • {feature:20s}: {value:.3f}")
        
        # Análise de conteúdo (se adaptativo)
        if self.adaptive_fusion:
            complexity = features['complexity']
            saturation = features['saturation']
            symmetry = features['symmetry']
            roughness = features['texture_roughness']
            
            semantic_score = (
                0.3 * complexity +
                0.3 * saturation +
                0.2 * (1 - symmetry) +
                0.2 * roughness
            )
            
            explanation.append(f"\n🎨 ANÁLISE DE CONTEÚDO:")
            explanation.append(f"  • Semantic Score: {semantic_score:.3f}")
            
            if semantic_score < 0.3:
                content_type = "ABSTRATO/FORMAL (minimalista, geométrico)"
            elif semantic_score < 0.6:
                content_type = "MISTO (elementos formais e narrativos)"
            else:
                content_type = "REPRESENTACIONAL (narrativo, detalhado)"
            
            explanation.append(f"  • Tipo de Conteúdo: {content_type}")
            explanation.append(f"  • Peso Neural Adaptado: α = {result['fusion_weight']:.3f}")
        
        # Fuzzy
        explanation.append("\n🧠 PREDIÇÃO FUZZY (regras interpretáveis):")
        fuzzy_sorted = sorted(
            result['fuzzy'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for emotion, prob in fuzzy_sorted[:3]:  # Top 3
            bar = "█" * int(prob * 30)
            explanation.append(f"  • {emotion:15s}: {prob:.3f} {bar}")
        
        # Neural (se disponível)
        if result['neural'] is not None:
            explanation.append("\n🤖 PREDIÇÃO NEURAL (SAT - conteúdo semântico):")
            neural_sorted = sorted(
                result['neural'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for emotion, prob in neural_sorted[:3]:
                bar = "█" * int(prob * 30)
                explanation.append(f"  • {emotion:15s}: {prob:.3f} {bar}")
        
        # Final
        fusion_info = f"α={result['fusion_weight']:.2f}"
        if self.adaptive_fusion:
            fusion_info += " (adaptativo)"
        
        explanation.append(f"\n🎯 PREDIÇÃO FINAL ({fusion_info}):")
        final_sorted = sorted(
            result['final'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for emotion, prob in final_sorted[:3]:
            bar = "█" * int(prob * 30)
            explanation.append(f"  • {emotion:15s}: {prob:.3f} {bar}")
        
        explanation.append("\n" + "=" * 70)
        
        return "\n".join(explanation)
    
    def compute_agreement(
        self,
        p_neural: Dict[str, float],
        p_fuzzy: Dict[str, float]
    ) -> float:
        """
        Calcula concordância entre neural e fuzzy (KL divergence).
        
        KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
        
        Quanto menor, maior a concordância.
        
        Args:
            p_neural: Distribuição neural
            p_fuzzy: Distribuição fuzzy
        
        Returns:
            KL divergence (0 = concordância total)
        """
        kl_div = 0.0
        
        for emotion in self.emotion_names:
            p = p_neural[emotion]
            q = p_fuzzy[emotion]
            
            # Evita log(0)
            if p > 1e-10 and q > 1e-10:
                kl_div += p * np.log(p / q)
        
        return kl_div


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    """Testa o sistema híbrido."""
    
    print("\n" + "="*70)
    print("TESTANDO SISTEMA HÍBRIDO NEURAL-FUZZY")
    print("="*70 + "\n")
    
    # Cria preditor (apenas fuzzy, sem SAT por enquanto)
    predictor = HybridEmotionPredictor(
        sat_checkpoint_path=None,  # TODO: adicionar quando SAT estiver pronto
        fusion_weight=0.7,
        use_guided_fuzzy=False
    )
    
    # Testa com uma imagem real
    test_image = "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg"
    
    print("=" * 70)
    print("TESTE: Franz Kline - Horizontal Rust (1960)")
    print("=" * 70)
    
    print(predictor.explain_prediction(test_image))
    
    print("\n" + "=" * 70)
    print("✅ SISTEMA HÍBRIDO FUNCIONANDO!")
    print("=" * 70 + "\n")
