"""
Regras Fuzzy adaptadas para features LAB

18 regras IF-THEN ajustadas para trabalhar com o espaço LAB perceptualmente uniforme.
"""

import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# FUNÇÕES DE PERTINÊNCIA (MEMBERSHIP FUNCTIONS) - LAB optimized
# ============================================================================

def low(x: float, threshold: float = 0.3) -> float:
    """Grau de pertinência LOW (otimizado para LAB)."""
    if x <= threshold:
        return 1.0
    elif x >= threshold + 0.2:
        return 0.0
    else:
        return 1.0 - (x - threshold) / 0.2


def medium(x: float, center: float = 0.5, width: float = 0.3) -> float:
    """Grau de pertinência MEDIUM (otimizado para LAB)."""
    if abs(x - center) <= width / 2:
        return 1.0
    elif abs(x - center) >= width:
        return 0.0
    else:
        return 1.0 - (abs(x - center) - width / 2) / (width / 2)


def high(x: float, threshold: float = 0.7) -> float:
    """Grau de pertinência HIGH (otimizado para LAB)."""
    if x >= threshold:
        return 1.0
    elif x <= threshold - 0.2:
        return 0.0
    else:
        return (x - (threshold - 0.2)) / 0.2


# ============================================================================
# REGRAS FUZZY (18 REGRAS) - LAB adapted
# ============================================================================

class LABFuzzyRules:
    """
    Sistema de 18 regras fuzzy adaptado para features LAB.
    
    Features LAB usadas:
    - brightness: L* direto (0-1)
    - color_temperature: a* normalizado (0=frio/verde, 1=quente/vermelho)
    - saturation: C* (chroma) normalizado
    - color_harmony: Entropia no círculo a*b*
    - complexity: Gradientes LAB
    - symmetry: Simetria em L*
    - texture_roughness: Variância de L*
    """
    
    # Mapeamento de emoções
    EMOTIONS = [
        'amusement', 'awe', 'contentment', 'excitement',
        'anger', 'disgust', 'fear', 'sadness', 'something else'
    ]
    
    def __init__(self):
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[Tuple]:
        """
        Define as 18 regras fuzzy adaptadas para LAB.
        
        Returns:
            Lista de tuplas (conditions, emotion, weight)
        """
        rules = [
            # REGRA 1: Excitement (alto brilho, quente, saturado)
            # LAB: L* alto + a* positivo (vermelho) + C* alto
            (
                lambda f: min(
                    high(f['brightness'], 0.65),
                    high(f['color_temperature'], 0.6),  # a* > 0 (vermelho)
                    high(f['saturation'], 0.6)
                ),
                'excitement',
                1.0
            ),
            
            # REGRA 2: Contentment (brilho médio, harmônico, suave)
            # LAB: L* médio + alta harmonia (entropia) + baixa rugosidade
            (
                lambda f: min(
                    medium(f['brightness'], 0.5, 0.4),
                    high(f['color_harmony'], 0.6),
                    low(f['texture_roughness'], 0.4)
                ),
                'contentment',
                1.0
            ),
            
            # REGRA 3: Amusement (cores vibrantes, complexo)
            # LAB: C* alto + complexidade alta + harmonia média
            (
                lambda f: min(
                    high(f['saturation'], 0.6),
                    high(f['complexity'], 0.6),
                    medium(f['color_harmony'], 0.5, 0.4)
                ),
                'amusement',
                0.9
            ),
            
            # REGRA 4: Awe (brilho alto, complexo, simétrico)
            # LAB: L* alto + gradientes altos + simetria
            (
                lambda f: min(
                    high(f['brightness'], 0.6),
                    high(f['complexity'], 0.5),
                    high(f['symmetry'], 0.5)
                ),
                'awe',
                1.0
            ),
            
            # REGRA 5: Sadness (escuro, frio, dessaturado)
            # LAB: L* baixo + a* negativo (verde/azul) + C* baixo
            (
                lambda f: min(
                    low(f['brightness'], 0.35),
                    low(f['color_temperature'], 0.4),  # a* < 0 (frio)
                    low(f['saturation'], 0.4)
                ),
                'sadness',
                1.0
            ),
            
            # REGRA 6: Fear (escuro, dessaturado, complexo)
            # LAB: L* baixo + C* baixo + gradientes altos
            (
                lambda f: min(
                    low(f['brightness'], 0.4),
                    low(f['saturation'], 0.45),
                    high(f['complexity'], 0.5)
                ),
                'fear',
                0.9
            ),
            
            # REGRA 7: Anger (quente intenso, rugoso)
            # LAB: a* muito positivo (vermelho) + C* alto + rugosidade alta
            (
                lambda f: min(
                    high(f['color_temperature'], 0.7),  # Vermelho forte
                    high(f['saturation'], 0.65),
                    high(f['texture_roughness'], 0.5)
                ),
                'anger',
                1.0
            ),
            
            # REGRA 8: Disgust (cores dissonantes, assimétrico)
            # LAB: Baixa harmonia (entropia) + assimetria + rugosidade
            (
                lambda f: min(
                    low(f['color_harmony'], 0.4),
                    low(f['symmetry'], 0.4),
                    high(f['texture_roughness'], 0.5)
                ),
                'disgust',
                0.8
            ),
            
            # REGRA 9: Contentment variant (tons pastéis)
            # LAB: L* alto + C* médio + harmonia alta
            (
                lambda f: min(
                    high(f['brightness'], 0.6),
                    medium(f['saturation'], 0.4, 0.3),
                    high(f['color_harmony'], 0.6)
                ),
                'contentment',
                0.8
            ),
            
            # REGRA 10: Awe variant (grandioso, brilho extremo)
            # LAB: L* muito alto + simetria + saturação média
            (
                lambda f: min(
                    high(f['brightness'], 0.75),
                    high(f['symmetry'], 0.6),
                    medium(f['saturation'], 0.5, 0.4)
                ),
                'awe',
                0.9
            ),
            
            # REGRA 11: Excitement variant (energia visual)
            # LAB: Gradientes altos + C* alto + a* positivo
            (
                lambda f: min(
                    high(f['complexity'], 0.65),
                    high(f['saturation'], 0.6),
                    high(f['color_temperature'], 0.55)
                ),
                'excitement',
                0.8
            ),
            
            # REGRA 12: Sadness variant (monocromático escuro)
            # LAB: L* baixo + C* muito baixo + harmonia baixa
            (
                lambda f: min(
                    low(f['brightness'], 0.3),
                    low(f['saturation'], 0.3),
                    low(f['color_harmony'], 0.45)
                ),
                'sadness',
                0.9
            ),
            
            # REGRA 13: Amusement variant (cores primárias)
            # LAB: C* muito alto + harmonia média (cores puras)
            (
                lambda f: min(
                    high(f['saturation'], 0.7),
                    medium(f['color_harmony'], 0.5, 0.3),
                    medium(f['complexity'], 0.5, 0.4)
                ),
                'amusement',
                0.85
            ),
            
            # REGRA 14: Fear variant (caótico escuro)
            # LAB: L* baixo + gradientes altos + assimetria
            (
                lambda f: min(
                    low(f['brightness'], 0.35),
                    high(f['complexity'], 0.6),
                    low(f['symmetry'], 0.4)
                ),
                'fear',
                0.85
            ),
            
            # REGRA 15: Anger variant (saturação intensa)
            # LAB: C* extremo + a* positivo + rugosidade
            (
                lambda f: min(
                    high(f['saturation'], 0.75),
                    high(f['color_temperature'], 0.65),
                    medium(f['texture_roughness'], 0.5, 0.3)
                ),
                'anger',
                0.85
            ),
            
            # REGRA 16: Disgust variant (tons esverdeados rugosos)
            # LAB: a* negativo (verde) + rugosidade + baixa harmonia
            (
                lambda f: min(
                    low(f['color_temperature'], 0.35),  # a* < 0 (verde)
                    high(f['texture_roughness'], 0.55),
                    low(f['color_harmony'], 0.45)
                ),
                'disgust',
                0.75
            ),
            
            # REGRA 17: Something else (features neutras)
            # LAB: Tudo médio (não se encaixa em outras categorias)
            (
                lambda f: min(
                    medium(f['brightness'], 0.5, 0.5),
                    medium(f['saturation'], 0.5, 0.5),
                    medium(f['complexity'], 0.5, 0.5)
                ),
                'something else',
                0.6
            ),
            
            # REGRA 18: Something else variant (ausência de padrão)
            # LAB: Baixa ativação em todas as regras anteriores
            (
                lambda f: 1.0 - max(
                    high(f['brightness'], 0.7),
                    low(f['brightness'], 0.3),
                    high(f['saturation'], 0.7),
                    low(f['saturation'], 0.3)
                ),
                'something else',
                0.5
            )
        ]
        
        return rules
    
    def infer(self, features: Dict[str, float]) -> np.ndarray:
        """
        Executa inferência fuzzy nas features LAB.
        
        Args:
            features: Dict com 7 features LAB normalizadas [0, 1]
        
        Returns:
            Array (9,) com probabilidades para cada emoção
        """
        # Inicializa scores
        emotion_scores = np.zeros(len(self.EMOTIONS))
        
        # Aplica cada regra
        for condition, emotion, weight in self.rules:
            # Avalia condição (grau de ativação)
            activation = condition(features)
            
            # Acumula score ponderado
            emotion_idx = self.EMOTIONS.index(emotion)
            emotion_scores[emotion_idx] += activation * weight
        
        # Normaliza para formar distribuição de probabilidade
        total = emotion_scores.sum()
        if total > 0:
            probabilities = emotion_scores / total
        else:
            # Fallback: distribuição uniforme
            probabilities = np.ones(len(self.EMOTIONS)) / len(self.EMOTIONS)
        
        return probabilities
    
    def infer_top_k(self, features: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
        """
        Retorna top-k emoções mais prováveis.
        
        Args:
            features: Dict com features LAB
            k: Número de emoções a retornar
        
        Returns:
            Lista de tuplas (emotion, probability) ordenadas
        """
        probs = self.infer(features)
        top_indices = np.argsort(probs)[::-1][:k]
        
        return [
            (self.EMOTIONS[idx], float(probs[idx]))
            for idx in top_indices
        ]
    
    def explain(self, features: Dict[str, float]) -> Dict:
        """
        Explica quais regras foram ativadas e por quê.
        
        Args:
            features: Dict com features LAB
        
        Returns:
            Dict com explicação detalhada
        """
        explanation = {
            'features': features,
            'activated_rules': [],
            'final_probabilities': {},
            'predicted_emotion': None
        }
        
        # Avalia cada regra
        for i, (condition, emotion, weight) in enumerate(self.rules):
            activation = condition(features)
            
            if activation > 0.1:  # Threshold de ativação significativa
                explanation['activated_rules'].append({
                    'rule_id': i + 1,
                    'emotion': emotion,
                    'activation': float(activation),
                    'weight': weight,
                    'contribution': float(activation * weight)
                })
        
        # Ordenar por contribuição
        explanation['activated_rules'].sort(
            key=lambda x: x['contribution'],
            reverse=True
        )
        
        # Calcula probabilidades finais
        probs = self.infer(features)
        for emotion, prob in zip(self.EMOTIONS, probs):
            explanation['final_probabilities'][emotion] = float(prob)
        
        # Predição final
        explanation['predicted_emotion'] = self.EMOTIONS[np.argmax(probs)]
        
        return explanation


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTE: LAB Fuzzy Rules")
    print("="*70)
    
    fuzzy = LABFuzzyRules()
    
    # Teste 1: Imagem alegre (brilho alto, quente, saturado)
    features_happy = {
        'brightness': 0.8,
        'color_temperature': 0.7,  # Quente (vermelho)
        'saturation': 0.75,
        'color_harmony': 0.6,
        'complexity': 0.5,
        'symmetry': 0.6,
        'texture_roughness': 0.3
    }
    
    print("\n📊 Teste 1: Features 'alegres'")
    print("-" * 70)
    for k, v in features_happy.items():
        print(f"  {k:20s}: {v:.2f}")
    
    print("\n🎯 Top-3 Emoções:")
    top_emotions = fuzzy.infer_top_k(features_happy, k=3)
    for emotion, prob in top_emotions:
        bar = "█" * int(prob * 30)
        print(f"  {emotion:15s}: {prob:.4f} {bar}")
    
    # Teste 2: Imagem triste (escuro, frio, dessaturado)
    features_sad = {
        'brightness': 0.25,
        'color_temperature': 0.3,  # Frio (azul/verde)
        'saturation': 0.2,
        'color_harmony': 0.4,
        'complexity': 0.3,
        'symmetry': 0.5,
        'texture_roughness': 0.4
    }
    
    print("\n📊 Teste 2: Features 'tristes'")
    print("-" * 70)
    for k, v in features_sad.items():
        print(f"  {k:20s}: {v:.2f}")
    
    print("\n🎯 Top-3 Emoções:")
    top_emotions = fuzzy.infer_top_k(features_sad, k=3)
    for emotion, prob in top_emotions:
        bar = "█" * int(prob * 30)
        print(f"  {emotion:15s}: {prob:.4f} {bar}")
    
    print("\n" + "="*70)
    print("✅ LAB Fuzzy Rules funcionando!")
    print("="*70)
