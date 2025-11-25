"""
============================================================================
FUZZY VARIABLES - MEMBERSHIP FUNCTIONS
============================================================================
Define as VARIÁVEIS FUZZY e suas FUNÇÕES DE PERTINÊNCIA (Membership Functions).

TEORIA:
-------
Uma variável fuzzy transforma um valor numérico CRISP em graus de pertinência
a múltiplos conjuntos fuzzy simultaneamente.

Exemplo:
    brightness = 0.25 (valor crisp)
    
    Após fuzzificação:
    {
        'muito_escuro': 0.75,  # 75% pertence a "muito escuro"
        'escuro': 0.25,        # 25% pertence a "escuro"
        'médio': 0.0,
        'claro': 0.0,
        'muito_claro': 0.0
    }

MEMBERSHIP FUNCTIONS USADAS:
----------------------------
Usamos funções TRIANGULARES (trimf) porque:
1. Simples de interpretar
2. Computacionalmente eficientes
3. Padrão em sistemas fuzzy Mamdani

Formato: trimf(x, [a, b, c])
- a: ponto onde começa (μ=0)
- b: ponto do pico (μ=1)
- c: ponto onde termina (μ=0)

VARIÁVEIS IMPLEMENTADAS:
------------------------
Para cada uma das 7 features visuais, definimos 5 termos linguísticos:
- muito_baixo / muito_escuro / muito_frio (etc)
- baixo / escuro / frio
- médio / neutro
- alto / claro / quente
- muito_alto / muito_claro / muito_quente

REFERÊNCIAS:
------------
- Zadeh (1965): Fuzzy Sets
- Mamdani & Assilian (1975): Fuzzy Logic Controller
- scikit-fuzzy documentation
============================================================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, List, Tuple


class FuzzyVariables:
    """
    Define todas as variáveis fuzzy do sistema.
    
    Esta classe cria as variáveis fuzzy de INPUT (features visuais)
    e OUTPUT (emoções) com suas respectivas membership functions.
    
    Attributes:
        input_vars: Dict com as 7 variáveis de input
        output_vars: Dict com as 9 variáveis de output (emoções)
    """
    
    def __init__(self):
        """Inicializa e cria todas as variáveis fuzzy."""
        self.input_vars = self._create_input_variables()
        self.output_vars = self._create_output_variables()
    
    # ========================================================================
    # INPUT VARIABLES (Features Visuais)
    # ========================================================================
    
    def _create_input_variables(self) -> Dict[str, ctrl.Antecedent]:
        """
        Cria as 7 variáveis fuzzy de INPUT (features visuais).
        
        Cada variável tem:
        - Universe: [0, 1] (valores normalizados)
        - 5 termos linguísticos com membership functions triangulares
        
        Returns:
            Dict com as variáveis fuzzy de input
        """
        # Universe compartilhado: [0, 1]
        universe = np.arange(0, 1.01, 0.01)
        
        # ====================================================================
        # 1. BRIGHTNESS (Brilho)
        # ====================================================================
        brightness = ctrl.Antecedent(universe, 'brightness')
        brightness['muito_escuro'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        brightness['escuro'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        brightness['medio'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        brightness['claro'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        brightness['muito_claro'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 2. COLOR TEMPERATURE (Temperatura de Cor)
        # ====================================================================
        color_temp = ctrl.Antecedent(universe, 'color_temperature')
        color_temp['muito_frio'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        color_temp['frio'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        color_temp['neutro'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        color_temp['quente'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        color_temp['muito_quente'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 3. SATURATION (Saturação)
        # ====================================================================
        saturation = ctrl.Antecedent(universe, 'saturation')
        saturation['muito_dessaturado'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        saturation['dessaturado'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        saturation['medio_saturado'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        saturation['saturado'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        saturation['muito_saturado'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 4. COLOR HARMONY (Harmonia Cromática)
        # ====================================================================
        harmony = ctrl.Antecedent(universe, 'color_harmony')
        harmony['muito_dissonante'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        harmony['dissonante'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        harmony['neutro'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        harmony['harmonico'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        harmony['muito_harmonico'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 5. COMPLEXITY (Complexidade Visual)
        # ====================================================================
        complexity = ctrl.Antecedent(universe, 'complexity')
        complexity['muito_simples'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        complexity['simples'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        complexity['medio'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        complexity['complexo'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        complexity['muito_complexo'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 6. SYMMETRY (Simetria)
        # ====================================================================
        symmetry = ctrl.Antecedent(universe, 'symmetry')
        symmetry['muito_assimetrico'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        symmetry['assimetrico'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        symmetry['levemente_simetrico'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        symmetry['simetrico'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        symmetry['muito_simetrico'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        # ====================================================================
        # 7. TEXTURE ROUGHNESS (Aspereza de Textura)
        # ====================================================================
        roughness = ctrl.Antecedent(universe, 'texture_roughness')
        roughness['muito_liso'] = fuzz.trimf(universe, [0.0, 0.0, 0.25])
        roughness['liso'] = fuzz.trimf(universe, [0.0, 0.25, 0.5])
        roughness['medio'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
        roughness['aspero'] = fuzz.trimf(universe, [0.5, 0.75, 1.0])
        roughness['muito_aspero'] = fuzz.trimf(universe, [0.75, 1.0, 1.0])
        
        return {
            'brightness': brightness,
            'color_temperature': color_temp,
            'saturation': saturation,
            'color_harmony': harmony,
            'complexity': complexity,
            'symmetry': symmetry,
            'texture_roughness': roughness
        }
    
    # ========================================================================
    # OUTPUT VARIABLES (Emoções)
    # ========================================================================
    
    def _create_output_variables(self) -> Dict[str, ctrl.Consequent]:
        """
        Cria as 9 variáveis fuzzy de OUTPUT (emoções do ArtEmis).
        
        Cada emoção tem 3 termos linguísticos:
        - baixo: pouca intensidade da emoção
        - medio: intensidade moderada
        - alto: alta intensidade
        
        Returns:
            Dict com as variáveis fuzzy de output (emoções)
        """
        # Universe compartilhado: [0, 1] (intensidade da emoção)
        universe = np.arange(0, 1.01, 0.01)
        
        # Lista das 9 emoções do dataset ArtEmis
        emotions = [
            'amusement',      # Diversão
            'awe',            # Admiração
            'contentment',    # Contentamento
            'excitement',     # Excitação
            'anger',          # Raiva
            'disgust',        # Nojo
            'fear',           # Medo
            'sadness',        # Tristeza
            'something_else'  # Outra emoção
        ]
        
        output_vars = {}
        
        for emotion in emotions:
            # Cria a variável de output
            var = ctrl.Consequent(universe, emotion)
            
            # Define 3 termos linguísticos para intensidade
            var['baixo'] = fuzz.trimf(universe, [0.0, 0.0, 0.5])
            var['medio'] = fuzz.trimf(universe, [0.0, 0.5, 1.0])
            var['alto'] = fuzz.trimf(universe, [0.5, 1.0, 1.0])
            
            output_vars[emotion] = var
        
        return output_vars
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_input_variable(self, name: str) -> ctrl.Antecedent:
        """Retorna uma variável de input pelo nome."""
        if name not in self.input_vars:
            raise ValueError(f"Variável de input '{name}' não existe!")
        return self.input_vars[name]
    
    def get_output_variable(self, name: str) -> ctrl.Consequent:
        """Retorna uma variável de output pelo nome."""
        if name not in self.output_vars:
            raise ValueError(f"Variável de output '{name}' não existe!")
        return self.output_vars[name]
    
    def list_input_variables(self) -> List[str]:
        """Lista os nomes de todas as variáveis de input."""
        return list(self.input_vars.keys())
    
    def list_output_variables(self) -> List[str]:
        """Lista os nomes de todas as variáveis de output."""
        return list(self.output_vars.keys())
    
    def get_terms(self, variable_name: str) -> List[str]:
        """
        Retorna os termos linguísticos de uma variável.
        
        Args:
            variable_name: Nome da variável (input ou output)
        
        Returns:
            Lista com os nomes dos termos (ex: ['muito_escuro', 'escuro', ...])
        """
        # Tenta input primeiro
        if variable_name in self.input_vars:
            return list(self.input_vars[variable_name].terms.keys())
        # Depois output
        elif variable_name in self.output_vars:
            return list(self.output_vars[variable_name].terms.keys())
        else:
            raise ValueError(f"Variável '{variable_name}' não existe!")


# ============================================================================
# FUNÇÃO AUXILIAR PARA FUZZIFICAÇÃO MANUAL
# ============================================================================

def fuzzify_value(value: float, variable: ctrl.Antecedent) -> Dict[str, float]:
    """
    Fuzzifica um valor numérico em graus de pertinência.
    
    Útil para DEBUG e testes - mostra exatamente quanto cada termo
    linguístico é ativado para um dado valor.
    
    Args:
        value: Valor numérico crisp [0, 1]
        variable: Variável fuzzy (Antecedent)
    
    Returns:
        Dict com graus de pertinência: {'termo': pertinência, ...}
    
    Example:
        >>> vars = FuzzyVariables()
        >>> brightness_var = vars.get_input_variable('brightness')
        >>> fuzzify_value(0.25, brightness_var)
        {
            'muito_escuro': 0.0,
            'escuro': 1.0,   # <-- 100% pertence a "escuro"
            'medio': 0.0,
            'claro': 0.0,
            'muito_claro': 0.0
        }
    """
    memberships = {}
    
    for term_name, term_mf in variable.terms.items():
        # Calcula o grau de pertinência usando a MF do termo
        membership_value = fuzz.interp_membership(
            variable.universe,
            term_mf.mf,
            value
        )
        memberships[term_name] = float(membership_value)
    
    return memberships


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    """Testa a criação das variáveis fuzzy."""
    
    print("\n" + "="*70)
    print("TESTANDO CRIAÇÃO DE VARIÁVEIS FUZZY")
    print("="*70 + "\n")
    
    # Cria as variáveis
    fuzzy_vars = FuzzyVariables()
    
    # Lista inputs
    print("📥 VARIÁVEIS DE INPUT (Features Visuais):")
    for var_name in fuzzy_vars.list_input_variables():
        terms = fuzzy_vars.get_terms(var_name)
        print(f"  • {var_name:20s} → {len(terms)} termos: {terms}")
    
    print()
    
    # Lista outputs
    print("📤 VARIÁVEIS DE OUTPUT (Emoções):")
    for var_name in fuzzy_vars.list_output_variables():
        terms = fuzzy_vars.get_terms(var_name)
        print(f"  • {var_name:20s} → {len(terms)} termos: {terms}")
    
    print("\n" + "="*70)
    print("✅ VARIÁVEIS CRIADAS COM SUCESSO!")
    print("="*70 + "\n")
    
    # Teste de fuzzificação
    print("🧪 TESTE DE FUZZIFICAÇÃO:")
    print("-" * 70)
    
    brightness_var = fuzzy_vars.get_input_variable('brightness')
    
    test_values = [0.15, 0.25, 0.5, 0.75, 0.9]
    
    for val in test_values:
        memberships = fuzzify_value(val, brightness_var)
        print(f"\nBrilho = {val:.2f}:")
        for term, degree in memberships.items():
            if degree > 0.01:  # Mostra só os relevantes
                print(f"  {term:20s}: {degree:.3f}")
    
    print("\n" + "="*70 + "\n")
