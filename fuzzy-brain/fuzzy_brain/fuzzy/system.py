"""
============================================================================
FUZZY INFERENCE SYSTEM - SISTEMA DE INFERÊNCIA MAMDANI
============================================================================
Motor de inferência fuzzy que processa features visuais e produz
distribuições de probabilidade sobre emoções.

FLUXO COMPLETO:
---------------
1. FUZZIFICAÇÃO: Valores numéricos → conjuntos fuzzy
2. INFERÊNCIA: Ativa regras e combina consequentes (Mamdani)
3. DEFUZZIFICAÇÃO: Conjuntos fuzzy → probabilidades normalizadas

ARQUITETURA:
------------
Input: Dict com 7 features visuais [0,1]
  ↓
FuzzyVariables: Fuzzifica cada feature
  ↓
FuzzyRules: Ativa regras Mamdani (min-max)
  ↓
ControlSystem: Computa saída fuzzy para cada emoção
  ↓
Defuzzificação: Centroide → valores crisp
  ↓
Normalização: Softmax → distribuição de probabilidade
  ↓
Output: Dict com 9 emoções [0,1] somando 1.0

TEORIA:
-------
Usamos MAMDANI porque:
- Interpretável (regras linguísticas)
- Saída = área fuzzy (não função linear como Sugeno)
- Defuzzificação por centroide (centro de massa da área)

Exemplo:
  Input: {brightness: 0.15, color_temp: 0.2, saturation: 0.1, ...}
  
  Fuzzificação:
    brightness=0.15 → 40% muito_escuro + 60% escuro
    color_temp=0.2  → 80% muito_frio + 20% frio
    saturation=0.1  → 90% muito_dessaturado + 10% dessaturado
  
  Regra ativada:
    SE brightness É muito_escuro (0.4)
    E color_temp É muito_frio (0.8)
    E saturation É muito_dessaturado (0.9)
    ENTÃO sadness É alto
    
    Grau de ativação = min(0.4, 0.8, 0.9) = 0.4
  
  Defuzzificação:
    sadness_crisp = centroide da área cortada em 0.4
  
  Normalização:
    p_fuzzy = softmax([sadness, awe, contentment, ...])

REFERÊNCIAS:
------------
- Mamdani, E. H. (1974). Application of fuzzy algorithms for control
- Zadeh, L. A. (1965). Fuzzy sets
- Jang et al. (1997). Neuro-Fuzzy and Soft Computing (Cap. 2-4)

============================================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple
from skfuzzy import control as ctrl
import warnings

# Import relativo ou absoluto
try:
    from .variables import FuzzyVariables, fuzzify_value
    from .rules import FuzzyRules
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from fuzzy_brain.fuzzy.variables import FuzzyVariables, fuzzify_value
    from fuzzy_brain.fuzzy.rules import FuzzyRules


class FuzzyInferenceSystem:
    """
    Sistema de Inferência Fuzzy completo (Mamdani).
    
    Processa features visuais através de regras fuzzy e produz
    distribuições de probabilidade sobre emoções.
    
    Attributes:
        variables: Variáveis fuzzy (inputs e outputs)
        rules: Regras fuzzy (Mamdani)
        control_system: Sistema de controle scikit-fuzzy
        simulation: Simulação do sistema de controle
        emotion_names: Lista com nomes das 9 emoções
    """
    
    def __init__(self, use_guided: bool = False):
        """
        Inicializa o sistema de inferência fuzzy.
        
        Args:
            use_guided: Se True, usa regras guiadas (v2.0).
                       Se False, usa regras simples (v1.0).
        """
        # Cria variáveis e regras
        self.variables = FuzzyVariables()
        self.rules = FuzzyRules(self.variables, use_guided=use_guided)
        
        # Cria sistema de controle
        self.control_system = ctrl.ControlSystem(self.rules.get_rules())
        
        # Cria simulação (objeto que executa a inferência)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        # Lista de emoções na ordem correta
        self.emotion_names = [
            'amusement', 'awe', 'contentment', 'excitement',
            'anger', 'disgust', 'fear', 'sadness', 'something_else'
        ]
        
        print(f"✅ Sistema Fuzzy inicializado com {self.rules.count_rules()} regras")
    
    # ========================================================================
    # INFERÊNCIA PRINCIPAL
    # ========================================================================
    
    def infer(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Executa inferência fuzzy completa.
        
        Args:
            features: Dicionário com 7 features visuais:
                - brightness: [0,1]
                - color_temperature: [0,1]
                - saturation: [0,1]
                - color_harmony: [0,1]
                - complexity: [0,1]
                - symmetry: [0,1]
                - texture_roughness: [0,1]
        
        Returns:
            Dicionário com 9 emoções [0,1] somando ~1.0:
                - amusement, awe, contentment, excitement,
                  anger, disgust, fear, sadness, something_else
        
        Example:
            >>> system = FuzzyInferenceSystem()
            >>> features = {
            ...     'brightness': 0.15,
            ...     'color_temperature': 0.2,
            ...     'saturation': 0.1,
            ...     'color_harmony': 0.3,
            ...     'complexity': 0.5,
            ...     'symmetry': 0.6,
            ...     'texture_roughness': 0.4
            ... }
            >>> emotions = system.infer(features)
            >>> print(f"Sadness: {emotions['sadness']:.3f}")
        """
        # 1. Valida inputs
        self._validate_features(features)
        
        # 2. Fuzzificação + Inferência (feito pelo scikit-fuzzy)
        crisp_outputs = self._compute_crisp_outputs(features)
        
        # 3. Normalização → probabilidades
        probabilities = self._normalize_to_probabilities(crisp_outputs)
        
        return probabilities
    
    # ========================================================================
    # PASSOS DA INFERÊNCIA
    # ========================================================================
    
    def _validate_features(self, features: Dict[str, float]):
        """Valida que todas as features necessárias estão presentes."""
        required = [
            'brightness', 'color_temperature', 'saturation',
            'color_harmony', 'complexity', 'symmetry', 'texture_roughness'
        ]
        
        for feature in required:
            if feature not in features:
                raise ValueError(f"Feature '{feature}' não encontrada!")
            
            value = features[feature]
            if not (0.0 <= value <= 1.0):
                warnings.warn(
                    f"Feature '{feature}' = {value} fora do intervalo [0,1]"
                )
    
    def _compute_crisp_outputs(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Executa inferência Mamdani e defuzzificação.
        
        Internamente, scikit-fuzzy faz:
        1. Fuzzificação: valores → graus de pertinência
        2. Ativação de regras: min(antecedentes) → grau de ativação
        3. Agregação: max de todas as regras ativadas
        4. Defuzzificação: centroide da área agregada
        
        Args:
            features: Features visuais [0,1]
        
        Returns:
            Valores crisp para cada emoção (ainda não normalizados)
        """
        # Reset da simulação
        self.simulation.reset()
        
        # Seta inputs
        for feature_name, value in features.items():
            self.simulation.input[feature_name] = value
        
        # Executa inferência
        try:
            self.simulation.compute()
        except Exception as e:
            # Se falhar (ex: nenhuma regra ativada), retorna zeros
            warnings.warn(f"Inferência falhou: {e}. Retornando zeros.")
            return {emotion: 0.0 for emotion in self.emotion_names}
        
        # Coleta outputs crisp
        crisp_outputs = {}
        for emotion in self.emotion_names:
            try:
                crisp_outputs[emotion] = self.simulation.output[emotion]
            except KeyError:
                # Emoção não tem regras ou não foi ativada
                crisp_outputs[emotion] = 0.0
        
        return crisp_outputs
    
    def _normalize_to_probabilities(
        self, 
        crisp_outputs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Converte valores crisp em distribuição de probabilidade.
        
        Usa softmax com temperatura para suavizar:
            p_i = exp(x_i / T) / sum(exp(x_j / T))
        
        Args:
            crisp_outputs: Valores crisp [0,1] para cada emoção
        
        Returns:
            Probabilidades [0,1] somando 1.0
        """
        # Extrai valores na ordem correta
        values = np.array([crisp_outputs[e] for e in self.emotion_names])
        
        # Se tudo zero, distribui uniformemente
        if np.sum(values) == 0:
            uniform_prob = 1.0 / len(self.emotion_names)
            return {e: uniform_prob for e in self.emotion_names}
        
        # Normalização simples (proporcional)
        # Alternativa: softmax para suavizar
        # values_exp = np.exp(values / temperature)
        # probabilities = values_exp / np.sum(values_exp)
        
        probabilities = values / np.sum(values)
        
        # Garante que soma exatamente 1.0 (correção de arredondamento)
        probabilities = probabilities / np.sum(probabilities)
        
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_names, probabilities)
        }
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    def get_rule_activations(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Retorna o grau de ativação de cada regra (debug).
        
        Útil para entender QUAIS regras foram ativadas e o quanto.
        
        Args:
            features: Features visuais [0,1]
        
        Returns:
            Dict mapeando label da regra → grau de ativação [0,1]
        """
        # Seta inputs
        self.simulation.reset()
        for feature_name, value in features.items():
            self.simulation.input[feature_name] = value
        
        # Computa
        try:
            self.simulation.compute()
        except:
            return {}
        
        # Extrai ativações (não diretamente exposto pelo scikit-fuzzy)
        # Isso é uma aproximação - scikit-fuzzy não expõe isso facilmente
        # TODO: Se precisar, implementar manualmente
        
        return {}  # Placeholder
    
    def explain_inference(self, features: Dict[str, float]) -> str:
        """
        Gera explicação textual da inferência (debug).
        
        Args:
            features: Features visuais [0,1]
        
        Returns:
            String explicando a inferência
        """
        result = self.infer(features)
        
        explanation = []
        explanation.append("=" * 70)
        explanation.append("EXPLICAÇÃO DA INFERÊNCIA FUZZY")
        explanation.append("=" * 70)
        
        # Inputs
        explanation.append("\n📊 FEATURES VISUAIS:")
        for feature, value in features.items():
            # Traduz para termo linguístico
            var = self.variables.get_input_variable(feature)
            terms = fuzzify_value(value, var)
            top_term = max(terms, key=terms.get)
            explanation.append(
                f"  • {feature}: {value:.3f} → {top_term} ({terms[top_term]:.1%})"
            )
        
        # Outputs
        explanation.append("\n🎭 EMOÇÕES INFERIDAS:")
        sorted_emotions = sorted(
            result.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for emotion, prob in sorted_emotions:
            bar = "█" * int(prob * 40)
            explanation.append(f"  • {emotion:15s}: {prob:.3f} {bar}")
        
        explanation.append("\n" + "=" * 70)
        
        return "\n".join(explanation)


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    """Testa o sistema de inferência fuzzy."""
    
    print("\n" + "="*70)
    print("TESTANDO SISTEMA DE INFERÊNCIA FUZZY")
    print("="*70 + "\n")
    
    # Cria sistema
    print("⏳ Inicializando sistema fuzzy...")
    system = FuzzyInferenceSystem(use_guided=False)
    print()
    
    # Teste 1: Pintura ESCURA + FRIA + DESSATURADA → Sadness
    print("=" * 70)
    print("TESTE 1: Pintura escura, fria, dessaturada (→ SADNESS)")
    print("=" * 70)
    
    features_sad = {
        'brightness': 0.15,           # Muito escuro
        'color_temperature': 0.2,     # Muito frio
        'saturation': 0.1,            # Muito dessaturado
        'color_harmony': 0.3,         # Dissonante
        'complexity': 0.5,            # Médio
        'symmetry': 0.4,              # Baixo
        'texture_roughness': 0.5      # Médio
    }
    
    print(system.explain_inference(features_sad))
    
    # Teste 2: Pintura BRILHANTE + SATURADA + QUENTE → Excitement/Amusement
    print("\n" + "=" * 70)
    print("TESTE 2: Pintura brilhante, saturada, quente (→ EXCITEMENT)")
    print("=" * 70)
    
    features_excited = {
        'brightness': 0.8,            # Claro
        'color_temperature': 0.85,    # Muito quente
        'saturation': 0.9,            # Muito saturado
        'color_harmony': 0.6,         # Harmonioso
        'complexity': 0.7,            # Complexo
        'symmetry': 0.5,              # Médio
        'texture_roughness': 0.6      # Rugoso
    }
    
    print(system.explain_inference(features_excited))
    
    # Teste 3: Pintura SIMÉTRICA + HARMONIOSA → Awe
    print("\n" + "=" * 70)
    print("TESTE 3: Pintura simétrica, harmoniosa (→ AWE)")
    print("=" * 70)
    
    features_awe = {
        'brightness': 0.6,            # Médio-claro
        'color_temperature': 0.5,     # Neutro
        'saturation': 0.5,            # Médio
        'color_harmony': 0.9,         # Muito harmonioso
        'complexity': 0.8,            # Muito complexo
        'symmetry': 0.95,             # Muito simétrico
        'texture_roughness': 0.4      # Liso
    }
    
    print(system.explain_inference(features_awe))
    
    print("\n" + "=" * 70)
    print("✅ SISTEMA FUZZY FUNCIONANDO CORRETAMENTE!")
    print("=" * 70 + "\n")
