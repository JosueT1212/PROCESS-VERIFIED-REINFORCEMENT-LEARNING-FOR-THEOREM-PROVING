import torch
import numpy as np
from lean_dojo_v2.lean_dojo import Dojo, Theorem, ProofFinished, LeanError, TimeoutError , TacticState
# ==========================================
# Environment class for Lean 4 theorem proving
# ==========================================

class LeanGRPOEnv:
    def __init__(self, repo, list_of_theorems,d2,d1):
        """
        Constructor: Prepara el entorno con los teoremas del dataset.
        No recibe tácticas, solo inicializa el contexto de Lean 4.
        """
        self.repo = repo
        self.d2 = d2
        self.d1 = d1
        self.theorems = list_of_theorems
        self.current_dojo = None
        self.current_state = None
        self.response = None
        
    def reset(self, theorem_idx,response):
        """
        Resetea el entorno para un nuevo episodio con el teorema especificado.
        Carga el teorema en el Dojo y devuelve el estado inicial.
        """
        theorem = self.theorems[theorem_idx]
        self.response = response
        self.current_dojo = Dojo(self.repo, theorem)
        self.current_state = self.current_dojo.initial_state()
        return self.current_state
    def parse_tactics(self, response):
        """
        Función para parsear la respuesta del modelo en una lista de tácticas.
        Asume que las tácticas están separadas por saltos de línea.
        """
        tactics = response.strip().split('\n')
        return [tactic.strip() for tactic in tactics if tactic.strip()]
    
        

    # ==========================================
    # Functions for reward calculation and credit assignment (Math 1, 2, 3)
    # ==========================================
            
    def score_function(self,estado_inicial ):
        tactics = self.parse_tactics(self.response)
        estado_actual = estado_inicial      
        for k, tactic in enumerate(tactics):
            result = self.current_dojo.run_tac(estado_actual, tactic.strip())
            
        
            if  isinstance(result, (LeanError,TimeoutError)):
                return 0
                
            else:
                estado_actual = result 
       
        if isinstance(result, ProofFinished):
            return 1.0
        else:
            return 0.0
    
                
    
    def per_tactic_score_function(self,estado_inicial):
        scores = []
        tactics = self.parse_tactics(self.response)
        estado_actual = estado_inicial   
    
        for k, tactic in enumerate(tactics):
            dojo_result = self.current_dojo.run_tac(estado_actual, tactic.strip())

            if isinstance(dojo_result, (LeanError, TimeoutError)):
                scores.append(self.d2)
            
            else:
                scores.append(self.d1)
            estado_actual = dojo_result

        if isinstance(dojo_result, ProofFinished):
            scores = [1.0] * len(tactics)

        return scores
    
    def outcome_reward_function(self,estado_inicial):
        if self.score_function(estado_inicial) == 1.0:
            return 1.0
        else:
            return 0.0
            
    def process_reward_function(self, estado_inicial):
            tactics = self.parse_tactics(self.response)
            estado_actual = estado_inicial
            rewards = []
        
            # CORRECCIÓN: Inicializar variables antes del bucle
            j = len(tactics) 
            dojo_result = None 
        
            for k, tactic in enumerate(tactics):
                # CORRECCIÓN: self.current_dojo
                dojo_result = self.current_dojo.run_tac(estado_actual, tactic.strip())

                if isinstance(dojo_result, (LeanError, TimeoutError)):
                    j = k
                    break
                else:
                    rewards.append(self.d1)
                    estado_actual = dojo_result

            # Validamos que dojo_result exista por si la lista tactics estaba vacía
            if dojo_result and isinstance(dojo_result, ProofFinished):
                rewards = [1.0] * len(tactics)
            # Si se ha producido un error, asignamos d2 a las tácticas restantes
            elif j < len(tactics):
                rewards.extend([self.d2] * (len(tactics) - j))
                
            return rewards
            
# ==========================================
# Auxiliary functions for advantages calculation
# ==========================================
  



    
def A_outcome_function(scores,len_tokens):
    '''
    scores: lista de recompensas por táctica (shape: G,)
    len_tokens: número de tokens en la secuencia de tácticas (shape: seq_length
    '''
    
    
    g_Y = torch.tensor(scores, dtype=torch.float32) # shape: (G,)
    
    
    mean_g = g_Y.mean()
    std_g = g_Y.std() + 1e-8 
    
    # 3. Aplicamos LA FÓRMULA DE TU IMAGEN
    
    A_outcome = (g_Y - mean_g) / std_g # shape: (G,)
    

    # Pasamos de shape (G,) -> (G, 1) -> (G, seq_length)
    A_outcome_tokens = A_outcome.unsqueeze(1).expand(-1, len_tokens)
    
    return A_outcome_tokens,mean_g

import torch

def get_first_token_mask(input_ids, newline_token_id):
    """
    input_ids: Tensor de PyTorch con shape (Batch_Size, Seq_Len)
    Devuelve: Una máscara booleana del mismo tamaño, donde True significa "aquí empieza una táctica".
    """
   
    first_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
   
    first_token_mask[:, 0] = True
    
    
    is_newline = (input_ids == newline_token_id)
    
    first_token_mask[:, 1:] = is_newline[:, :-1]
    
    return first_token_mask

def A_process_function(sample_responses_tokens, rewards_list, mean_g, newline_token_id, len_tokens):
    '''
    sample_responses_tokens: Tensor con los tokens generados (shape: G, len_tokens)
    rewards_list: Lista de G listas con recompensas por TÁCTICA. Ej: [[d1, d1], [d1, d2, d2]]
    mean_g: Float con la media de A_outcome
    newline_token_id: Int
    len_tokens: Int
    '''
    G = len(rewards_list)
    
   
    token_rewards = torch.zeros((G, len_tokens), dtype=torch.float32)
    
    mask = get_first_token_mask(sample_responses_tokens, newline_token_id=newline_token_id)
    
    
    for i in range(G):
        
        tactic_start_indices = mask[i].nonzero(as_tuple=True)[0]
        
        num_tactics = min(len(rewards_list[i]), len(tactic_start_indices))
        
       
        for t in range(num_tactics):
            idx = tactic_start_indices[t]
            token_rewards[i, idx] = rewards_list[i][t]
            
    
    A_process_tokens = (token_rewards - mean_g) * mask
    
    return A_process_tokens