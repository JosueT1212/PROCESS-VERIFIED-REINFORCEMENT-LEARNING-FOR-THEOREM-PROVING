import torch
import torch.nn.functional as F
import random
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from env.env import LeanGRPOEnv, A_outcome_function, A_process_function, get_first_token_mask

class GRPOLoRAModel:
    def __init__(self, model_name, repo, theorems, d1=0.1, d2=-1.0, G=8):
        """
        Inicializa el modelo base, LoRA, el tokenizador y los hiperparámetros.
        """
        self.repo = repo
        self.all_theorems = theorems
        self.d1 = d1
        self.d2 = d2
        self.G = G  # Tamaño del grupo de respuestas
        self.epsilon = 0.2  # Para el clipping de GRPO
        self.beta = 0.01  # Penalización KL
        
        # 1. Configurar Tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Cargar Modelo Base en bfloat16 para ahorrar memoria
        print("Cargando modelo base...")
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 3. Inyectar LoRA
        print("Configurando LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.policy_model = get_peft_model(model_base, lora_config)
        self.policy_model.print_trainable_parameters()
        
        # 4. Configurar Optimizador (Solo actualiza los pesos de LoRA)
        self.optimizer = AdamW(self.policy_model.parameters(), lr=1e-5)
        
        # Variables para los splits
        self.train_theorems = []
        self.test_theorems = []

    def train_test_split(self, train_ratio=0.8):
        """
        Divide el dataset de teoremas en conjunto de entrenamiento y prueba.
        """
        random.shuffle(self.all_theorems)
        split_idx = int(len(self.all_theorems) * train_ratio)
        
        self.train_theorems = self.all_theorems[:split_idx]
        self.test_theorems = self.all_theorems[split_idx:]
        
        print(f"Dataset dividido: {len(self.train_theorems)} para entrenar, {len(self.test_theorems)} para testear.")

    def train(self, num_epochs=1):
        """
        Bucle principal de entrenamiento usando GRPO.
        """
        self.policy_model.train()
        # Inicializamos el entorno con los teoremas de entrenamiento
        env = LeanGRPOEnv(self.repo, self.train_theorems, self.d2, self.d1)
        
        for epoch in range(num_epochs):
            print(f"--- Iniciando Epoch {epoch + 1}/{num_epochs} ---")
            
            for theorem_idx in range(len(self.train_theorems)):
                prompt_text = f"Resuelve este teorema en Lean 4. Escribe solo las tácticas, una por línea:\n{self.train_theorems[theorem_idx]}"
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
                prompt_length = inputs["input_ids"].shape[1]
                
                # 1. Generar G respuestas
                with torch.no_grad():
                    generations = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_return_sequences=self.G,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response_tokens = generations[:, prompt_length:]
                len_tokens = response_tokens.shape[1]
                
                scores_outcome = []
                rewards_process_list = []
                
                # 2. Evaluar en el entorno Lean
                for i in range(self.G):
                    response_text = self.tokenizer.decode(response_tokens[i], skip_special_tokens=True)
                    estado_inicial = env.reset(theorem_idx, response_text)
                    
                    scores_outcome.append(env.outcome_reward_function(estado_inicial))
                    rewards_process_list.append(env.process_reward_function(estado_inicial))
                    
                # 3. Calcular Ventajas (Tus funciones)
                A_out_tokens, mean_g = A_outcome_function(scores_outcome, len_tokens)
                A_out_tokens = A_out_tokens.to("cuda")
                
                A_proc_tokens = A_process_function(
                    response_tokens, rewards_process_list, mean_g, self.tokenizer.encode('\n')[0], len_tokens
                ).to("cuda")
                
                A_total = A_out_tokens + A_proc_tokens
                
                # 4. Optimización (Actualizar pesos)
                self.optimizer.zero_grad()
                
                # Probabilidades de Policy (Con LoRA)
                logits_policy = self.policy_model(generations).logits[:, prompt_length-1:-1, :]
                logprobs_policy = F.log_softmax(logits_policy, dim=-1)
                token_logprobs_policy = torch.gather(logprobs_policy, 2, response_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Probabilidades de Referencia (Sin LoRA - ¡El truco mágico!)
                with torch.no_grad():
                    with self.policy_model.disable_adapter():
                        logits_ref = self.policy_model(generations).logits[:, prompt_length-1:-1, :]
                        logprobs_ref = F.log_softmax(logits_ref, dim=-1)
                        token_logprobs_ref = torch.gather(logprobs_ref, 2, response_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Divergencia KL y Ratio
                log_ratio = token_logprobs_policy - token_logprobs_ref
                ratio = torch.exp(log_ratio)
                kl_div = torch.exp(-log_ratio) + log_ratio - 1.0
                
                # Pérdida (Loss)
                surr1 = ratio * A_total
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * A_total
                loss = -torch.mean(torch.min(surr1, surr2) - self.beta * kl_div)
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                print(f"Teorema {theorem_idx} entrenado. Loss: {loss.item():.4f} | Éxitos: {sum(scores_outcome)}/{self.G}")

    def test(self):
        """
        Evalúa el modelo en el conjunto de prueba (Test Set).
        """
        print("\n--- Iniciando Evaluación en Test Set ---")
        self.policy_model.eval() 
        env = LeanGRPOEnv(self.repo, self.test_theorems, self.d2, self.d1)
        
        teoremas_resueltos = 0
        
        for theorem_idx in range(len(self.test_theorems)):
            prompt_text = f"Resuelve este teorema en Lean 4. Escribe solo las tácticas, una por línea:\n{self.test_theorems[theorem_idx]}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
            
            
            with torch.no_grad():
                generations = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=1,
                    do_sample=False, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            prompt_length = inputs["input_ids"].shape[1]
            response_tokens = generations[0, prompt_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            
            estado_inicial = env.reset(theorem_idx, response_text)
            
            
            score = env.score_function(estado_inicial)
            
            if score == 1.0:
                teoremas_resueltos += 1
                print(f"Teorema {theorem_idx}: ¡RESUELTO!")
            else:
                print(f"Teorema {theorem_idx}: Falló.")
                
        accuracy = (teoremas_resueltos / len(self.test_theorems)) * 100
        print(f"\nResultados del Test: {teoremas_resueltos}/{len(self.test_theorems)} resueltos ({accuracy:.2f}%)")
        return accuracy