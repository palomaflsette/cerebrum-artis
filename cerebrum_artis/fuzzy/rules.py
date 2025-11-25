"""
============================================================================
FUZZY RULES - REGRAS MAMDANI PARA EMOÇÕES EM ARTE
============================================================================
Define as REGRAS FUZZY que mapeiam features visuais → emoções.

TEORIA:
-------
Regras fuzzy são a PONTE entre features interpretáveis e emoções.
Cada regra é baseada em conhecimento especialista (psicologia das cores,
teoria da estética) e tem a forma:

    SE <antecedentes> ENTÃO <consequente>
    
Exemplo:
    SE brightness É muito_escuro
    E color_temperature É frio
    E saturation É dessaturado
    ENTÃO sadness É alto

FUNDAMENTAÇÃO CIENTÍFICA:
--------------------------
Cada regra está fundamentada em literatura de psicologia das cores:

1. SADNESS (Tristeza):
   - Valdez & Mehrabian (1994): Cores escuras, frias, dessaturadas
   - Itten (1970): Azul escuro associado a melancolia

2. AWE (Admiração):
   - Palmer & Schloss (2010): Simetria e harmonia evocam admiração
   - Ramachandran & Hirstein (1999): Simetria ativa centros de prazer

3. CONTENTMENT (Contentamento):
   - Elliot & Maier (2007): Cores suaves, médias, equilibradas
   - Warm colors em baixa intensidade → conforto

4. EXCITEMENT (Excitação):
   - Ou et al. (2018): Cores saturadas, quentes, complexas
   - Alta energia visual → arousal

5. ANGER (Raiva):
   - Fetterman et al. (2011): Vermelho e alta saturação
   - Cores quentes intensas → agressividade

E assim por diante...

ESTRATÉGIA DE IMPLEMENTAÇÃO:
-----------------------------
Versão 1.0 (AGORA): Regras simples Mamdani
  - 15-20 regras cobrindo as 9 emoções
  - Sem uso de p_neural (guided)
  - Foco em robustez e interpretabilidade

Versão 2.0 (DEPOIS - se der tempo): Regras guiadas
  - Hedges adaptativos usando p_neural
  - Análise de concordância/conflito
  - Contribuição científica original

============================================================================
"""

import numpy as np
from skfuzzy import control as ctrl
from typing import Dict, List

# Import relativo ou absoluto dependendo de como é chamado
try:
    from .variables import FuzzyVariables
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from fuzzy_brain.fuzzy.variables import FuzzyVariables


class FuzzyRules:
    """
    Define e gerencia as regras fuzzy do sistema.
    
    Esta classe cria regras Mamdani que mapeiam features visuais
    (brightness, saturation, etc) para emoções (sadness, awe, etc).
    
    Attributes:
        variables: Instância de FuzzyVariables
        rules: Lista de regras fuzzy (ctrl.Rule)
        use_guided: Flag para usar regras guiadas (v2.0)
    """
    
    def __init__(self, variables: FuzzyVariables, use_guided: bool = False):
        """
        Inicializa o sistema de regras.
        
        Args:
            variables: Objeto com variáveis fuzzy de input/output
            use_guided: Se True, usa regras guiadas por p_neural (v2.0)
                       Na v2.0, usa as mesmas regras simples mas com
                       modulação neural via _apply_neural_guidance()
        """
        self.variables = variables
        self.use_guided = use_guided
        
        # Cria as regras (sempre as mesmas, guided modula depois)
        self.rules = self._create_simple_rules()
    
    # ========================================================================
    # REGRAS SIMPLES (v1.0)
    # ========================================================================
    
    def _create_simple_rules(self) -> List[ctrl.Rule]:
        """
        Cria as regras fuzzy SIMPLES (versão básica Mamdani).
        
        Total de regras: ~18 (2-3 por emoção principal)
        
        Returns:
            Lista de objetos ctrl.Rule
        """
        rules = []
        
        # Atalhos para variáveis (facilita escrita das regras)
        brightness = self.variables.input_vars['brightness']
        color_temp = self.variables.input_vars['color_temperature']
        saturation = self.variables.input_vars['saturation']
        harmony = self.variables.input_vars['color_harmony']
        complexity = self.variables.input_vars['complexity']
        symmetry = self.variables.input_vars['symmetry']
        roughness = self.variables.input_vars['texture_roughness']
        
        # Emoções
        sadness = self.variables.output_vars['sadness']
        awe = self.variables.output_vars['awe']
        contentment = self.variables.output_vars['contentment']
        excitement = self.variables.output_vars['excitement']
        anger = self.variables.output_vars['anger']
        fear = self.variables.output_vars['fear']
        disgust = self.variables.output_vars['disgust']
        amusement = self.variables.output_vars['amusement']
        
        # ====================================================================
        # SADNESS (Tristeza) - 3 regras
        # ====================================================================
        
        # Regra 1: Cores escuras, frias, dessaturadas → Tristeza
        # Ref: Valdez & Mehrabian (1994)
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['muito_escuro'] &
                color_temp['muito_frio'] &
                saturation['muito_dessaturado']
            ),
            consequent=sadness['alto'],
            label='SADNESS_1_dark_cold_desat'
        ))
        
        # Regra 2: Escuro + frio (mais geral)
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['escuro'] &
                color_temp['frio']
            ),
            consequent=sadness['medio'],
            label='SADNESS_2_dark_cold'
        ))
        
        # Regra 3: Baixa saturação + baixa harmonia → Melancolia
        rules.append(ctrl.Rule(
            antecedent=(
                saturation['dessaturado'] &
                harmony['muito_dissonante']
            ),
            consequent=sadness['medio'],
            label='SADNESS_3_desat_dissonant'
        ))
        
        # ====================================================================
        # AWE (Admiração) - 3 regras
        # ====================================================================
        
        # Regra 4: Alta simetria + harmonia → Admiração
        # Ref: Ramachandran & Hirstein (1999)
        rules.append(ctrl.Rule(
            antecedent=(
                symmetry['muito_simetrico'] &
                harmony['muito_harmonico']
            ),
            consequent=awe['alto'],
            label='AWE_1_symmetry_harmony'
        ))
        
        # Regra 5: Complexidade alta + harmonia → Admiração pela beleza complexa
        rules.append(ctrl.Rule(
            antecedent=(
                complexity['muito_complexo'] &
                harmony['harmonico']
            ),
            consequent=awe['medio'],
            label='AWE_2_complex_harmony'
        ))
        
        # Regra 6: Simetria + brilho claro → Admiração
        rules.append(ctrl.Rule(
            antecedent=(
                symmetry['simetrico'] &
                brightness['muito_claro']
            ),
            consequent=awe['medio'],
            label='AWE_3_symmetry_bright'
        ))
        
        # ====================================================================
        # CONTENTMENT (Contentamento) - 2 regras
        # ====================================================================
        
        # Regra 7: Cores suaves, equilibradas → Contentamento
        # Ref: Elliot & Maier (2007)
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['medio'] &
                saturation['medio_saturado'] &
                color_temp['neutro']
            ),
            consequent=contentment['alto'],
            label='CONTENTMENT_1_balanced'
        ))
        
        # Regra 8: Harmonia + simplicidade → Serenidade
        rules.append(ctrl.Rule(
            antecedent=(
                harmony['harmonico'] &
                complexity['simples']
            ),
            consequent=contentment['medio'],
            label='CONTENTMENT_2_harmony_simple'
        ))
        
        # ====================================================================
        # EXCITEMENT (Excitação) - 3 regras
        # ====================================================================
        
        # Regra 9: Cores saturadas, quentes, complexas → Excitação
        # Ref: Ou et al. (2018)
        rules.append(ctrl.Rule(
            antecedent=(
                saturation['muito_saturado'] &
                color_temp['muito_quente'] &
                complexity['complexo']
            ),
            consequent=excitement['alto'],
            label='EXCITEMENT_1_sat_warm_complex'
        ))
        
        # Regra 10: Alta saturação + quente
        rules.append(ctrl.Rule(
            antecedent=(
                saturation['saturado'] &
                color_temp['quente']
            ),
            consequent=excitement['medio'],
            label='EXCITEMENT_2_sat_warm'
        ))
        
        # Regra 11: Complexidade + textura áspera → Energia visual
        rules.append(ctrl.Rule(
            antecedent=(
                complexity['muito_complexo'] &
                roughness['muito_aspero']
            ),
            consequent=excitement['medio'],
            label='EXCITEMENT_3_complex_rough'
        ))
        
        # ====================================================================
        # ANGER (Raiva) - 2 regras
        # ====================================================================
        
        # Regra 12: Vermelho intenso (quente + saturado) → Raiva
        # Ref: Fetterman et al. (2011)
        rules.append(ctrl.Rule(
            antecedent=(
                color_temp['muito_quente'] &
                saturation['muito_saturado'] &
                roughness['aspero']
            ),
            consequent=anger['alto'],
            label='ANGER_1_red_intense'
        ))
        
        # Regra 13: Dissonância + cores quentes
        rules.append(ctrl.Rule(
            antecedent=(
                harmony['muito_dissonante'] &
                color_temp['quente']
            ),
            consequent=anger['medio'],
            label='ANGER_2_dissonant_warm'
        ))
        
        # ====================================================================
        # FEAR (Medo) - 2 regras
        # ====================================================================
        
        # Regra 14: Escuro + assimétrico + dissonante → Medo
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['muito_escuro'] &
                symmetry['muito_assimetrico'] &
                harmony['dissonante']
            ),
            consequent=fear['alto'],
            label='FEAR_1_dark_asymm_dissonant'
        ))
        
        # Regra 15: Escuro + frio + complexo
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['escuro'] &
                color_temp['muito_frio'] &
                complexity['muito_complexo']
            ),
            consequent=fear['medio'],
            label='FEAR_2_dark_cold_complex'
        ))
        
        # ====================================================================
        # AMUSEMENT (Diversão) - 2 regras
        # ====================================================================
        
        # Regra 16: Cores vivas, quentes, claras → Diversão
        rules.append(ctrl.Rule(
            antecedent=(
                brightness['claro'] &
                saturation['saturado'] &
                color_temp['quente']
            ),
            consequent=amusement['alto'],
            label='AMUSEMENT_1_bright_sat_warm'
        ))
        
        # Regra 17: Alta saturação + harmonia
        rules.append(ctrl.Rule(
            antecedent=(
                saturation['muito_saturado'] &
                harmony['harmonico']
            ),
            consequent=amusement['medio'],
            label='AMUSEMENT_2_sat_harmony'
        ))
        
        # ====================================================================
        # DISGUST (Nojo) - 1 regra
        # ====================================================================
        
        # Regra 18: Dissonante + textura áspera + escuro → Nojo
        rules.append(ctrl.Rule(
            antecedent=(
                harmony['muito_dissonante'] &
                roughness['muito_aspero'] &
                brightness['escuro']
            ),
            consequent=disgust['medio'],
            label='DISGUST_1_dissonant_rough_dark'
        ))
        
        return rules
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_rules(self) -> List[ctrl.Rule]:
        """Retorna a lista de regras."""
        return self.rules
    
    def count_rules(self) -> int:
        """Retorna o número total de regras."""
        return len(self.rules)
    
    def list_rules(self) -> List[str]:
        """Lista os nomes (labels) de todas as regras."""
        return [rule.label for rule in self.rules]
    
    def get_rules_by_emotion(self, emotion: str) -> List[ctrl.Rule]:
        """
        Filtra regras por emoção.
        
        Args:
            emotion: Nome da emoção (ex: 'sadness', 'awe')
        
        Returns:
            Lista de regras que predizem essa emoção
        """
        emotion_upper = emotion.upper()
        return [
            rule for rule in self.rules 
            if rule.label.startswith(emotion_upper)
        ]
    
    def print_rule_summary(self):
        """Imprime um resumo das regras por emoção."""
        emotions = [
            'sadness', 'awe', 'contentment', 'excitement',
            'anger', 'fear', 'disgust', 'amusement'
        ]
        
        print("\n" + "="*70)
        print("RESUMO DAS REGRAS FUZZY")
        print("="*70 + "\n")
        
        for emotion in emotions:
            rules = self.get_rules_by_emotion(emotion)
            print(f"{emotion.upper():15s}: {len(rules)} regras")
            for rule in rules:
                print(f"  • {rule.label}")
        
        print(f"\n{'TOTAL':15s}: {self.count_rules()} regras")
        print("="*70 + "\n")


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    """Testa a criação das regras."""
    
    # Adiciona path para imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("\n" + "="*70)
    print("TESTANDO CRIAÇÃO DE REGRAS FUZZY")
    print("="*70 + "\n")
    
    # Cria variáveis
    print("⏳ Criando variáveis fuzzy...")
    fuzzy_vars = FuzzyVariables()
    print("✅ Variáveis criadas!\n")
    
    # Cria regras
    print("⏳ Criando regras Mamdani...")
    fuzzy_rules = FuzzyRules(fuzzy_vars, use_guided=False)
    print(f"✅ {fuzzy_rules.count_rules()} regras criadas!\n")
    
    # Mostra resumo
    fuzzy_rules.print_rule_summary()
    
    # Mostra exemplos de regras específicas
    print("📋 EXEMPLOS DE REGRAS DETALHADAS:")
    print("-" * 70)
    
    # Sadness
    print("\n🔵 SADNESS (Tristeza):")
    for rule in fuzzy_rules.get_rules_by_emotion('sadness'):
        print(f"  {rule.label}")
        print(f"    {rule}")
    
    # Awe
    print("\n✨ AWE (Admiração):")
    for rule in fuzzy_rules.get_rules_by_emotion('awe'):
        print(f"  {rule.label}")
        print(f"    {rule}")
    
    print("\n" + "="*70)
    print("✅ REGRAS CRIADAS E VALIDADAS COM SUCESSO!")
    print("="*70 + "\n")
