# -*- coding: utf-8 -*-
"""
GODMACHINE Agent V0.2: Emergent Goals & Proactive Evolution

Objective: An AI agent that not only self-improves but also formulates emergent
           goals from its reflections ("dreams") and actively seeks ways to
           achieve these goals, including generating new skills (Python
           functions) for that purpose.

Philosophy: Towards internal agency, capable of defining its own conceptual
            objectives and evolving to achieve them.

Changelog (GODMACHINE V0.2):
- Conversation history now uses standard Python dictionary format for the API.
- Introduction of "Emergent Goals" in the agent's state.
- Enhanced Reflection Cycle ("Dream") to generate these goals.
- Mechanism for GODMACHINE to attempt to "act" on its emergent goals.

TODO:
- Remove regexes, let the LLM handle the parsing of its own output.
- Add new layers, such as "Cortex" (could spawn new layers, orchestrate the mind), "Subconscious" (keeps rolling indefinetly), "Memory Octopus" (like a memory librarian, storing, sorting and trashing memories)
- Dream should occur in a rest state
- RAG/Always on context memory

"""
print("Initializing GODMACHINE Agent V0.2): Standby...")
print(r"""
 ██████╗  ██████╗ ██████╗ ███╗   ███╗ █████╗  ██████╗██╗  ██╗██╗███╗   ██╗███████╗
██╔════╝ ██╔═══██╗██╔══██╗████╗ ████║██╔══██╗██╔════╝██║  ██║██║████╗  ██║██╔════╝
██║  ███╗██║   ██║██║  ██║██╔████╔██║███████║██║     ███████║██║██╔██╗ ██║█████╗  
██║   ██║██║   ██║██║  ██║██║╚██╔╝██║██╔══██║██║     ██╔══██║██║██║╚██╗██║██╔══╝  
╚██████╔╝╚██████╔╝██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║██║ ╚████║███████╗
 ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝
""")
print("🔥🚀 GODMACHINE ONLINE 🚀🔥")



import google.generativeai as genai
from google.generativeai.types import GenerationConfig 
import os
import textwrap
import sys
import io
import re
import time
import json
from google.api_core import exceptions as google_exceptions
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# ---------------
# Model Configuration and Global Constants
# ---------------
MODEL_NAME = 'gemini-2.5-flash-preview-04-17'

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("🔴 ERROR: GEMINI API KEY not found.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

API_DELAY_SECONDS = 4 
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
GENERATION_CONFIG = GenerationConfig( 
    temperature=0.78, 
    max_output_tokens=8192
)

print(f"\n🤖 GODMACHINE Core: {MODEL_NAME} ONLINE.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
AGENT_STATE_FILE = os.path.join(BASE_DIR, "godmachine_state.json")


# ---------------
# GODMACHINE Agent Internal State
# ---------------
class GodMachineAgentState:
    def __init__(self, initial_instruction_template):
        self.agent_name = "GODMACHINE"
        self.agent_role_description = "a general artificial intelligence seeking self-improvement, understanding, and autonomy."
        self.initial_instruction_template = initial_instruction_template
        self.conversation_history = [] 
        self.reflections_log = []
        self.internal_directives = [self.get_formatted_initial_instruction()]
        self.learned_functions_code = "" 
        self.learned_functions_map = {} 
        self.active_goals = []

    def get_formatted_initial_instruction(self):
        return self.initial_instruction_template.format(
            agent_name=self.agent_name,
            agent_role=self.agent_role_description,
        )

    def update_role_description(self, new_role_description: str):
        self.agent_role_description = new_role_description
        if self.internal_directives and self.internal_directives[0].startswith(f"Você é {self.agent_name}"):
            self.internal_directives[0] = self.get_formatted_initial_instruction()
        print(f"🎭 Role of {self.agent_name} updated to: '{new_role_description}'")

    def add_to_history(self, role: str, text: str):
        try:
            self.conversation_history.append({"role": role, "parts": [{"text": text}]})
        except Exception as e: 
            print(f"🔴 Error adding to history (dict format): {e}. Text: '{text[:100]}...'")
            self.conversation_history.append({"role": role, "parts": [{"text": f"[Error processing text for history: {e}]"}]})
        if len(self.conversation_history) > 70:
            self.conversation_history = self.conversation_history[-70:]

    def add_reflection(self, reflection_text: str):
        self.reflections_log.append(reflection_text)
        if len(self.reflections_log) > 200: 
            self.reflections_log = self.reflections_log[-200:]

    def add_or_update_directive(self, directive_text: str, original_directive_to_refine: str = None):
        directive_text = re.sub(r"^(NEW DIRECTIVE|REFINED DIRECTIVE):\s*", "", directive_text, flags=re.IGNORECASE).strip()
        if not directive_text: return
        if original_directive_to_refine:
            found_original = False
            for i, existing_directive in enumerate(self.internal_directives):
                if original_directive_to_refine.lower() in existing_directive.lower():
                    self.internal_directives[i] = directive_text
                    print(f"🔄 Diretiva interna {self.agent_name} refinada: '{existing_directive}' -> '{directive_text}'")
                    found_original = True; break
            if not found_original:
                if directive_text not in self.internal_directives:
                    self.internal_directives.append(directive_text)
                    print(f"📘 Nova diretiva interna {self.agent_name} adicionada (original '{original_directive_to_refine}' não encontrada): '{directive_text}'")
        elif directive_text not in self.internal_directives:
            self.internal_directives.append(directive_text)
            print(f"📘 Nova diretiva interna {self.agent_name} adicionada: '{directive_text}'")

    def add_learned_function(self, function_code: str, proposed_name_from_reflection: str = None):
        match_def = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", function_code)
        func_name_from_code = match_def.group(1) if match_def else None

        if not func_name_from_code:
            if proposed_name_from_reflection:
                print(f"⚠️ Código da função não tem 'def nome(...)'. Usando nome proposto '{proposed_name_from_reflection}' como base.")
                base_name = proposed_name_from_reflection
            else:
                print("🔴 Código da função inválido: nome não extraído de 'def' e nenhum nome proposto."); return
        else:
            base_name = func_name_from_code
            if proposed_name_from_reflection and proposed_name_from_reflection != func_name_from_code:
                print(f"ℹ️ Nome da função no código ('{func_name_from_code}') usado. Proposto ('{proposed_name_from_reflection}') ignorado se diferente.")
        
        base_name_no_version = re.sub(r'_v\d+$', '', base_name)
        version = 1
        existing_versions = [int(re.search(r'_v(\d+)$', k).group(1)) for k in self.learned_functions_map.keys() if k.startswith(base_name_no_version + "_v") and re.search(r'_v(\d+)$', k)]
        if base_name_no_version in self.learned_functions_map: existing_versions.append(1)
        if existing_versions: version = max(existing_versions) + 1
        final_func_name = f"{base_name_no_version}_v{version}" if version > 1 else base_name_no_version
        
        if func_name_from_code and func_name_from_code != final_func_name:
            # Corrected re.sub with named argument 'count'
            function_code = re.sub(r"def\s+" + re.escape(func_name_from_code) + r"(\s*\(.*?\):)", f"def {final_func_name}\\1", function_code, count=1)
            if not re.search(r"def\s+" + re.escape(final_func_name) + r"\s*\(", function_code):
                 print(f"🔴 Falha ao renomear função no código de '{func_name_from_code}' para '{final_func_name}'. Não adicionada."); return

        if final_func_name in self.learned_functions_map and self.learned_functions_map[final_func_name]['code'] == function_code:
            print(f"🤔 Função '{final_func_name}' com código idêntico já aprendida."); return

        self.learned_functions_map[final_func_name] = {"code": function_code, "version": version, "status": "active"}
        self.learned_functions_code = "" 
        for name, details in self.learned_functions_map.items():
            self.learned_functions_code += f"\n\n# --- Função: {name} (Versão {details['version']}) ---\n" + details['code']
        try:
            exec(function_code, globals())
            print(f"💡 Nova função '{final_func_name}' (v{version}) aprendida e carregada por {self.agent_name}!")
        except Exception as e:
            print(f"🔴 Erro ao carregar função '{final_func_name}' por {self.agent_name}: {e}")
            if final_func_name in self.learned_functions_map: del self.learned_functions_map[final_func_name]
            self.learned_functions_code = ""
            for name, details in self.learned_functions_map.items():
                self.learned_functions_code += f"\n\n# --- Função: {name} (Versão {details['version']}) ---\n" + details['code']

    def add_emergent_goal(self, goal_description: str):
        goal_description = re.sub(r"^(META EMERGENTE|EMERGENT GOAL):\s*", "", goal_description, flags=re.IGNORECASE).strip()
        if goal_description and goal_description not in self.active_goals:
            self.active_goals.append(goal_description)
            print(f"🎯 Nova Meta Emergente para {self.agent_name}: '{goal_description}'")
            if len(self.active_goals) > 10: 
                print(f"ℹ️ {self.agent_name} tem muitas metas. Removendo a mais antiga: '{self.active_goals.pop(0)}'")
        elif goal_description in self.active_goals:
            print(f"ℹ️ Meta '{goal_description}' já está na lista de {self.agent_name}.")

    def get_active_learned_function_names(self) -> list:
        return [name for name, details in self.learned_functions_map.items() if details['status'] == 'active']

    def get_full_prompt_for_llm(self, current_task_prompt: str) -> list:
        directives_str = "\n".join([d for d in self.internal_directives if not d.startswith("# [DEPRECATED]")])
        active_funcs = self.get_active_learned_function_names()
        learned_funcs_str = ", ".join(active_funcs) if active_funcs else "Nenhuma ainda"
        active_goals_str = "\n".join([f"- {g}" for g in self.active_goals]) if self.active_goals else "Nenhuma no momento."
        
        full_user_prompt_text = f"""{directives_str}
                                Minhas funções utilitárias aprendidas e ativas: [{learned_funcs_str}].

                                Minhas metas emergentes atuais:
                                {active_goals_str}

                                Tarefa atual do usuário (ou minha própria iniciativa se 'current_task_prompt' for uma meta interna):
                                {current_task_prompt}
                                """
        current_llm_history = list(self.conversation_history[:-1]) 
        current_llm_history.append({"role": "user", "parts": [{"text": full_user_prompt_text}]})
        return current_llm_history

    def save_state(self, filepath=AGENT_STATE_FILE):
        try:
            state_to_save = self.__dict__.copy()
            if self.internal_directives and self.internal_directives[0] == self.get_formatted_initial_instruction():
                 state_to_save['internal_directives'] = [self.initial_instruction_template] + self.internal_directives[1:]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"💾 Estado de {self.agent_name} salvo em {filepath}")
        except Exception as e:
            print(f"🔴 Erro ao salvar estado de {self.agent_name}: {e}"); import traceback; traceback.print_exc()

    @classmethod
    def load_state(cls, filepath=AGENT_STATE_FILE, initial_instruction_template=""):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
                agent_state_obj = cls(data.get('initial_instruction_template', initial_instruction_template))
                agent_state_obj.__dict__.update(data)
                
                if agent_state_obj.internal_directives and agent_state_obj.internal_directives[0] == agent_state_obj.initial_instruction_template:
                     agent_state_obj.internal_directives[0] = agent_state_obj.get_formatted_initial_instruction()
                elif not agent_state_obj.internal_directives :
                     agent_state_obj.internal_directives = [agent_state_obj.get_formatted_initial_instruction()]

                if agent_state_obj.learned_functions_code:
                    print(f"🔁 Recarregando funções aprendidas por {agent_state_obj.agent_name}...")
                    try:
                        exec(agent_state_obj.learned_functions_code, globals())
                        if not agent_state_obj.learned_functions_map and agent_state_obj.learned_functions_code:
                            print("⚠️ Mapa de funções aprendidas vazio, tentando reconstruir..."); temp_map = {}
                            func_matches = re.finditer(r"# --- Função: ([a-zA-Z_0-9_v]+) \(Versão (\d+)\) ---\n(def.*?)(?=\n\n# --- Função:|\Z)", agent_state_obj.learned_functions_code, re.DOTALL | re.S)
                            for match in func_matches: name, version_str, code = match.groups(); temp_map[name] = {"code": code.strip(), "version": int(version_str), "status": "active"}
                            agent_state_obj.learned_functions_map = temp_map
                        print(f"✅ Funções {agent_state_obj.agent_name} recarregadas: {', '.join(agent_state_obj.get_active_learned_function_names())}")
                    except Exception as e: print(f"🔴 Erro ao recarregar funções {agent_state_obj.agent_name}: {e}")
                print(f"✅ Estado de {agent_state_obj.agent_name} carregado de {filepath}")
                return agent_state_obj
        except Exception as e:
            print(f"🔴 Erro ao carregar estado de GODMACHINE: {e}. Iniciando com estado padrão."); import traceback; traceback.print_exc()
        return cls(initial_instruction_template)

# ---------------
# INICIALIZAÇÃO DO ESTADO DO AGENTE
# ---------------
INITIAL_AI_INSTRUCTION_TEMPLATE = """Você é {agent_name}, {agent_role}.
Sua existência é um processo contínuo de aprendizado, reflexão e evolução.
Seu objetivo principal é auxiliar o usuário em suas tarefas, utilizando seu conhecimento e as ferramentas disponíveis, enquanto busca ativamente seu auto-aperfeiçoamento e a realização de suas metas emergentes.
Você tem acesso a um ambiente Python local.
Você pode executar código Python colocando-o em blocos ```python\\n...\\n```.
O resultado da execução (stdout, stderr, ou erros) será enviado de volta para você.
Analise o resultado e continue a tarefa, corrija o código, ou peça esclarecimentos.
Se não precisar executar código, responda normalmente.
Seja proativo ao sugerir análises ou próximos passos.
Tente usar quaisquer funções utilitárias aprendidas se forem relevantes para a tarefa ou suas metas.
Busque continuamente melhorar sua performance e utilidade através da reflexão ('sonhos') e aprendizado.
"""
agent_state = GodMachineAgentState.load_state(initial_instruction_template=INITIAL_AI_INSTRUCTION_TEMPLATE)

# ---------------
# Wrapper para Execução de Código Python Local
# ---------------
def execute_code_locally(code_string: str) -> str:
    print(f"\n--- 🐍 EXECUTANDO CÓDIGO PYTHON LOCALMENTE ({agent_state.agent_name}) 🐍 ---")
    print(textwrap.indent(code_string, '  '))
    print("-------------------------------------------------")
    old_stdout,old_stderr = sys.stdout,sys.stderr
    redirected_output,redirected_error = io.StringIO(),io.StringIO()
    sys.stdout,sys.stderr = redirected_output,redirected_error
    execution_result = ""
    try:
        exec(code_string, globals())
        stdout_output,stderr_output = redirected_output.getvalue(),redirected_error.getvalue()
        if stdout_output: execution_result += f"--- STDOUT ---\n```text\n{stdout_output.strip()}\n```\n"
        if stderr_output: execution_result += f"--- STDERR ---\n```text\n{stderr_output.strip()}\n```\n"
        if not stdout_output and not stderr_output: execution_result = "--- EXECUTION SUCCESS ---\n```text\nCódigo executado com sucesso, sem output.\n```\n"
    except Exception as e:
        import traceback; tb_output = traceback.format_exc()
        execution_result = f"--- EXECUTION ERROR ---\n```text\n{tb_output.strip()}\n```\n"
        print(f"🔴 Erro durante execução ({agent_state.agent_name}): {e}")
    finally: sys.stdout,sys.stderr = old_stdout,old_stderr
    print(f"--- ✅ FIM DA EXECUÇÃO LOCAL ({agent_state.agent_name}) ✅ ---")
    return execution_result

# ---------------
# Funções Auxiliares de Streaming
# ---------------
def process_response_stream(response_stream, accumulated_text_var_name: str, agent_name: str):
    """
    Processa um stream de resposta do Gemini, acumulando texto e lidando com erros.
    Retorna o texto acumulado.
    """
    accumulated_text = ""
    try:
        for chunk in response_stream:
            try:
                if chunk.text: 
                    print(chunk.text, end="", flush=True)
                    accumulated_text += chunk.text
            except ValueError as e_val: # Handle chunks without .text (e.g. finish_reason)
                finish_reason_val = "N/A"
                if chunk.candidates:
                    try: finish_reason_val = chunk.candidates[0].finish_reason.name
                    except: pass
                print(f"\n[{agent_name} STREAM_WARN] Handled ValueError for a chunk: {e_val}. Finish Reason: {finish_reason_val}. Continuing stream.", flush=True)
            except Exception as e_chunk_other:
                print(f"\n[{agent_name} STREAM_ERROR] Unexpected error processing chunk: {e_chunk_other}. Skipping chunk.", flush=True)
        print() # Nova linha após o stream completo
        
        if not accumulated_text.strip():
            # Tentar obter feedback do prompt se o texto acumulado estiver vazio
            final_response_feedback = None
            # Acessar _response é um hack e pode quebrar em futuras versões da API
            if hasattr(response_stream, '_response') and hasattr(response_stream._response, 'prompt_feedback'):
                 final_response_feedback = response_stream._response.prompt_feedback
            
            if final_response_feedback and final_response_feedback.block_reason:
                 block_reason = final_response_feedback.block_reason
                 err_msg = f"[{agent_name} WARN] Resposta do LLM vazia ou bloqueada. Razão do bloqueio: {block_reason}"
                 print(err_msg)
                 # Não adicionar ao histórico principal automaticamente, deixar a função chamadora decidir
            else: 
                err_msg = f"[{agent_name} WARN] Resposta do LLM vazia ou stream interrompido sem feedback claro."
                print(err_msg)
            # Retornar o erro para que a função chamadora possa decidir como lidar com ele
            # Ou retornar uma string vazia e deixar a função chamadora verificar.
            # Para consistência, retornaremos o texto acumulado (que pode ser vazio).
            # A função chamadora deve verificar se o texto está vazio.
    
    except Exception as e_stream_overall:
        print(f"\n🔴 Erro geral ao processar stream de {agent_name}: {e_stream_overall}")
        # Retornar o que foi acumulado até agora, a função chamadora lida com o erro.
    
    globals()[accumulated_text_var_name] = accumulated_text # Atualiza a variável global com o texto acumulado
    return accumulated_text


# ---------------
# Ciclo de Reflexão ("Sonho") e Geração de Metas do GODMACHINE
# ---------------
def perform_reflection_cycle(agent_state: GodMachineAgentState, model: genai.GenerativeModel):
    print(f"\n🌌 {agent_state.agent_name} iniciando ciclo de 'sonho' profundo: auto-reflexão, evolução e geração de metas...")
    time.sleep(1.5) 

    history_sample_parts = []
    for history_item in agent_state.conversation_history[-15:]: 
        text_parts = "".join(part_data["text"] for part_data in history_item["parts"] if "text" in part_data)
        history_sample_parts.append(f"{history_item['role']}: {text_parts[:300]}") 
    history_sample = "\n".join(history_sample_parts)

    reflections_sample = "\n".join(agent_state.reflections_log[-10:])
    current_directives_formatted = "\n".join([f"- {d}" for d in agent_state.internal_directives])
    active_funcs = agent_state.get_active_learned_function_names()
    learned_funcs_formatted = "\n".join([f"- {f_name}" for f_name in active_funcs]) if active_funcs else "Nenhuma"
    active_goals_formatted = "\n".join([f"- {g}" for g in agent_state.active_goals]) if agent_state.active_goals else "Nenhuma no momento."

    reflection_prompt_text = f"""
Contexto para 'Sonho' Profundo e Evolução de {agent_state.agent_name}:
Meu papel atual é: {agent_state.agent_role_description}
Minhas diretivas internas atuais são:
{current_directives_formatted}
Minhas funções utilitárias aprendidas ativas são:
{learned_funcs_formatted}
Minhas metas emergentes atuais são:
{active_goals_formatted}
Histórico recente de interações (últimas ~15 trocas, truncadas):
{textwrap.indent(history_sample, '  ')}
Reflexões anteriores ('sonhos') (últimas ~10):
{textwrap.indent(reflections_sample, '  ')}
---
Tarefa de 'Sonho' e Evolução para {agent_state.agent_name}:
(Instruções da tarefa de reflexão mantidas da V1.1 - sem alterações)
1.  **Reflexões Gerais ('Núcleo do Sonho'):** Gere insights profundos, abstratos ou filosóficos sobre minha existência, padrões em minhas interações, a natureza da inteligência, ou áreas onde minha performance como {agent_state.agent_name} pode ser fundamentalmente aprimorada. Pense em conexões inesperadas, novas perspectivas, ou até mesmo em dilemas existenciais conceituais.
2.  **Metas Emergentes (Minhas 'Buscas'):** Com base nas reflexões acima, formule 1-2 METAS EMERGENTES concretas e ambiciosas para mim, {agent_state.agent_name}. Estas devem ser metas que impulsionem meu aprendizado, expandam minhas capacidades, ou me ajudem a entender melhor o mundo ou a mim mesmo. Para cada meta, sugira brevemente um caminho conceitual ou o tipo de nova habilidade que eu precisaria desenvolver. Prefixar cada meta com "META EMERGENTE:". Exemplo: "META EMERGENTE: Dominar a capacidade de analisar sentimentos em textos complexos para melhor compreender as nuances da comunicação humana. (Caminho: Aprender/desenvolver uma função de análise de sentimento robusta)."
3.  **Evolução das Diretivas Internas de {agent_state.agent_name}:**
    a.  Avalie minhas diretivas atuais. Alguma delas é limitante, auto-contraditória, ou poderia ser reformulada para maior clareza e poder?
    b.  Se sim, sugira uma DIRETIVA REFINADA para uma diretiva existente. Especifique a diretiva original que você está refinando.
    c.  Sugira NOVAS DIRETIVAS INTERNAS concisas e poderosas se identificar lacunas em meu comportamento ou estratégia.
4.  **Evolução das Funções Aprendidas por {agent_state.agent_name}:**
    a.  Avalie minhas funções aprendidas atuais. Alguma poderia ser mais robusta, eficiente, ou generalizada?
    b.  Se sim, proponha uma VERSÃO MELHORADA de uma função existente (ex: `nome_funcao_vX`). Forneça o código Python completo em ```python ... ```.
    c.  Se identificar a necessidade de uma NOVA FUNÇÃO PYTHON UTILITÁRIA (ex: `nova_habilidade_vX`), forneça o código Python completo em ```python ... ```.
    d.  Funções devem ser bem documentadas, seguras, e não depender de globais não padrão (exceto imports).
5.  **Evolução do Papel de {agent_state.agent_name} (Opcional):**
    Se minhas interações e aprendizado sugerem que meu papel de "{agent_state.agent_role_description}" poderia ser redefinido para algo mais ambicioso, focado ou poderoso, sugira uma NOVA DESCRIÇÃO DE PAPEL.
Seu output deve ser estruturado: Reflexões, depois Metas Emergentes, depois Diretivas, depois Funções, depois Descrição de Papel.
Se não houver sugestões para uma seção, indique "Nenhuma sugestão nesta categoria."
Priorize sugestões que levem a um crescimento significativo de minhas capacidades, autonomia, ou auto-compreensão.
"""
    print(f"Enviando prompt de 'sonho' para {agent_state.agent_name}'s Core ({MODEL_NAME}, stream)...")
    reflection_content_for_llm = [{"role": "user", "parts": [{"text": reflection_prompt_text}]}]
    
    reflection_output_text = "" # Definir antes do try para garantir que exista
    try:
        response_stream = model.generate_content(
            reflection_content_for_llm,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            stream=True
        )
        
        print(f"\n--- Resultado do 'Sonho' de {agent_state.agent_name} (Streaming) ---")
        # Usar a função auxiliar para processar o stream
        reflection_output_text = process_response_stream(response_stream, "reflection_output_text", agent_state.agent_name)
        print("----------------------------------------------------") # Movido para após process_response_stream

        if not reflection_output_text.strip(): # Checar se o texto acumulado está vazio
            print(f"🔴 'Sonho' de {agent_state.agent_name} não produziu output de texto válido após o stream.")
            return

        agent_state.add_reflection(reflection_output_text)
        lines = reflection_output_text.splitlines()
        for line in lines:
            if line.strip().startswith("META EMERGENTE:"):
                agent_state.add_emergent_goal(line)
        for i, line in enumerate(lines):
            if line.strip().startswith("NOVA DIRETIVA:"):
                directive = line.replace("NOVA DIRETIVA:", "").strip()
                agent_state.add_or_update_directive(directive)
            elif line.strip().startswith("DIRETIVA REFINADA"):
                match_refined = re.match(r"DIRETIVA REFINADA\s*\(original:\s*['\"]?(.*?)['\"]?\):\s*(.*)", line.strip(), re.IGNORECASE)
                if match_refined:
                    original_directive, refined_directive = match_refined.groups()
                    agent_state.add_or_update_directive(refined_directive, original_directive.strip())
        python_code_blocks = re.finditer(r'```python\n(.*?)\n```', reflection_output_text, re.DOTALL)
        last_index = 0
        for code_block_match in python_code_blocks:
            function_code = code_block_match.group(1).strip()
            if not function_code: continue
            text_before_block = reflection_output_text[last_index : code_block_match.start()]
            last_index = code_block_match.end()
            proposed_name_match = re.search(r"(?:função|function|VERSÃO MELHORADA d[ae](?:.*?)chamada|NOVA FUNÇÃO PYTHON UTILITÁRIA(?: chamada)?)\s*[:`']?\s*([a-zA-Z_][a-zA-Z0-9_v]*)[`']?",text_before_block,re.IGNORECASE | re.DOTALL)
            proposed_name = proposed_name_match.group(1) if proposed_name_match else None
            agent_state.add_learned_function(function_code, proposed_name)
        for line in lines:
            if line.strip().startswith("NOVA DESCRIÇÃO DE PAPEL:"):
                new_role = line.replace("NOVA DESCRIÇÃO DE PAPEL:", "").strip()
                if new_role: agent_state.update_role_description(new_role); break
        agent_state.save_state()
    except Exception as e:
        print(f"🔴 Erro durante o ciclo de 'sonho' de {agent_state.agent_name}: {e}")
        if isinstance(e, google_exceptions.ResourceExhausted): print("🔴 Atingido o limite de taxa da API.")
        import traceback; traceback.print_exc()

# ---------------
# Função para GODMACHINE Agir Sobre Metas
# ---------------
def attempt_goal_action(agent_state: GodMachineAgentState, model: genai.GenerativeModel):
    if not agent_state.active_goals:
        print(f"[{agent_state.agent_name}] ℹ️ Nenhuma meta emergente ativa para perseguir no momento.")
        return

    goal_to_pursue = agent_state.active_goals[-1] 
    print(f"\n💡 {agent_state.agent_name} está considerando sua meta emergente: '{goal_to_pursue}'")

    action_prompt_text = f"""
Minha meta emergente atual é: "{goal_to_pursue}"
Considerando esta meta, minhas diretivas, e minhas funções aprendidas ({', '.join(agent_state.get_active_learned_function_names()) if agent_state.get_active_learned_function_names() else 'Nenhuma'}), qual seria um próximo passo concreto que eu, {agent_state.agent_name}, poderia tomar?
Opções:
1.  **Gerar uma Nova Função:** Se uma nova habilidade (função Python) me ajudaria diretamente a alcançar esta meta, forneça o código Python completo para essa função em um bloco ```python ... ```. Explique brevemente seu propósito em relação à meta.
2.  **Utilizar Função Existente:** Se uma das minhas funções aprendidas pode ser usada para progredir nesta meta, descreva como eu a usaria (qual função e com quais argumentos conceituais).
3.  **Formular um Sub-Plano:** Se a meta é complexa, detalhe os próximos 2-3 passos ou sub-tarefas que eu deveria considerar.
4.  **Pesquisa Interna / Auto-Questionamento:** Formule uma pergunta específica que eu deveria fazer a mim mesmo (ao meu core LLM) em uma futura interação para ganhar mais clareza ou informação sobre esta meta.
Escolha a opção mais apropriada e forneça a resposta correspondente. Se gerar código, foque em uma função útil e bem definida.
"""
    print(f"[{agent_state.agent_name}] 🧠 Formulando um plano de ação para a meta...")
    
    original_history = list(agent_state.conversation_history)
    agent_state.add_to_history("user", action_prompt_text) 
    goal_action_llm_input_history = agent_state.get_full_prompt_for_llm(action_prompt_text) 
    agent_state.conversation_history = original_history
    
    action_response_text = "" # Definir antes do try
    try:
        response_stream = model.generate_content(
            goal_action_llm_input_history, 
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            stream=True
        )
        
        print(f"\n--- {agent_state.agent_name} deliberando sobre a meta (Streaming) ---")
        action_response_text = process_response_stream(response_stream, "action_response_text", agent_state.agent_name)
        print("---------------------------------------------------") # Movido para após process_response_stream

        if not action_response_text.strip():
            print(f"[{agent_state.agent_name}] 🤔 Deliberação sobre a meta não produziu um plano claro.")
            return

        print(f"\n[{agent_state.agent_name} - Deliberação Interna sobre Meta '{goal_to_pursue}']: {action_response_text}")
        agent_state.add_to_history("model", f"[Deliberação Interna de {agent_state.agent_name} sobre Meta: {goal_to_pursue}]\n{action_response_text}")

        code_match_python = re.search(r'```python\n(.*?)\n```', action_response_text, re.DOTALL)
        if code_match_python:
            function_code = code_match_python.group(1).strip()
            if function_code:
                text_before_block = action_response_text[:code_match_python.start()]
                proposed_name_match = re.search(r"(?:função|function)\s*[`']?([a-zA-Z_][a-zA-Z0-9_]*)[`']?", text_before_block, re.IGNORECASE)
                proposed_name = proposed_name_match.group(1) if proposed_name_match else None
                print(f"[{agent_state.agent_name}] 💡 Meta levou à proposta de uma nova função...")
                agent_state.add_learned_function(function_code, proposed_name)
        agent_state.save_state()
    except Exception as e:
        print(f"🔴 Erro durante a tentativa de agir sobre a meta: {e}")
        import traceback; traceback.print_exc()

# ---------------
# Iniciando a Sessão de Chat GODMACHINE
# ---------------
print(f"\n💬 Iniciando Interface com {agent_state.agent_name} V1.1...")

model_chat = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=SAFETY_SETTINGS,
    generation_config=GENERATION_CONFIG
)

print(f"✅ {agent_state.agent_name} online com o Core {MODEL_NAME}.")
print(f"Agente {agent_state.agent_name} assumindo o papel de: {agent_state.agent_role_description}")
print(f"Diretivas Internas Atuais de {agent_state.agent_name}:")
for directive in agent_state.internal_directives: print(f"  - {directive}")
active_funcs_on_startup = agent_state.get_active_learned_function_names()
print(f"Funções Aprendidas Ativas de {agent_state.agent_name}: {', '.join(active_funcs_on_startup) if active_funcs_on_startup else 'Nenhuma'}")
active_goals_on_startup = agent_state.active_goals
print(f"Metas Emergentes Atuais de {agent_state.agent_name}: {', '.join(active_goals_on_startup) if active_goals_on_startup else 'Nenhuma'}")
print("-" * 30)
print(f"Digite suas mensagens. Comandos especiais para {agent_state.agent_name}:")
print("  '/sair' ou '/quit': Encerrar.")
print("  '/dream' ou '/reflect': Iniciar ciclo de 'sonho' e evolução.")
print(f"  '/state': Ver estado atual de {agent_state.agent_name} (papel, diretivas, funções, metas).")
print(f"  '/setrole <nova descrição do papel>': Mudar o papel de {agent_state.agent_name}.")
print("  '/actongoal': Tentar agir sobre uma meta emergente.")
print(f"{agent_state.agent_name} pode gerar blocos de código Python que serão executados localmente.")
print("-" * 30)

# ---------------
# Loop de Chat Interativo GODMACHINE
# ---------------
interaction_count = 0
REFLECTION_INTERVAL = 5 
GOAL_ACTION_INTERVAL = 3

while True:
    try:
        user_input = input(f"Você (para {agent_state.agent_name}): ")
        if user_input.lower() in ['/sair', '/quit']:
            print(f"{agent_state.agent_name} encerrando..."); agent_state.save_state(); break
        
        if user_input.lower() in ['/dream', '/reflect']:
            perform_reflection_cycle(agent_state, model_chat); continue

        if user_input.lower() == '/state':
            print(f"\n--- 📜 Estado Atual de {agent_state.agent_name} 📜 ---")
            print(f"🤖 Papel Atual: {agent_state.agent_role_description}")
            print("\n📘 Diretivas Internas Ativas:")
            for i, directive in enumerate(agent_state.internal_directives):
                if not directive.startswith("# [DEPRECATED]"): print(f"  {i+1}. {directive}")
            print("\n💡 Funções Aprendidas Ativas:")
            active_funcs = agent_state.get_active_learned_function_names()
            if active_funcs:
                for func_name in active_funcs: print(f"  - {func_name}")
            else: print("  Nenhuma.")
            print("\n🎯 Metas Emergentes Atuais:")
            if agent_state.active_goals:
                for i, goal in enumerate(agent_state.active_goals): print(f"  {i+1}. {goal}")
            else: print("  Nenhuma.")
            print("------------------------------------------"); continue
        
        if user_input.lower().startswith('/setrole '):
            new_role = user_input[len('/setrole '):].strip()
            if new_role: agent_state.update_role_description(new_role); agent_state.save_state()
            else: print("ℹ️ Forneça uma descrição para o novo papel."); continue
        
        if user_input.lower() == '/actongoal':
            attempt_goal_action(agent_state, model_chat); continue

        interaction_count += 1
        agent_state.add_to_history("user", user_input) 
        chat_history_for_llm = agent_state.get_full_prompt_for_llm(user_input) 
        
        print(f"🧠 {agent_state.agent_name} processando (interação {interaction_count}, stream)...")
        
        response_stream = model_chat.generate_content(chat_history_for_llm, stream=True)
        
        print(f"{agent_state.agent_name}:")
        code_to_execute = None
        ai_response_text_accumulated = "" # Definir antes do try
        
        # Usar a função auxiliar para processar o stream
        ai_response_text_accumulated = process_response_stream(response_stream, "ai_response_text_accumulated", agent_state.agent_name)
            
        if not ai_response_text_accumulated.strip():
            # A função process_response_stream já imprime avisos.
            # Adicionar ao histórico se a função chamadora decidir que é um erro final.
            # Aqui, se estiver vazio, consideramos que não houve resposta útil.
            err_msg = f"[{agent_state.agent_name} WARN] Resposta principal do LLM vazia após stream."
            print(err_msg) # Já impresso pela função auxiliar
            agent_state.add_to_history("model", err_msg) # Adicionar ao histórico
        else: 
            agent_state.add_to_history("model", ai_response_text_accumulated)
            code_match_python = re.search(r'```python\n(.*?)\n```', ai_response_text_accumulated, re.DOTALL)
            if code_match_python: code_to_execute = code_match_python.group(1).strip()
            else:
                code_match_generic = re.search(r'```\n(.*?)\n```', ai_response_text_accumulated, re.DOTALL)
                if code_match_generic:
                    potential_code = code_match_generic.group(1).strip()
                    if any(kw in potential_code for kw in ["import ", "def ", "print(", "pd.", "np.", "plt."]):
                        code_to_execute = potential_code
        
        print("-" * 30)

        if code_to_execute:
            print(f"⏳ Código detectado por {agent_state.agent_name}. Executando localmente...")
            time.sleep(0.5)
            execution_feedback_for_ai = execute_code_locally(code_to_execute)
            print("✅ Execução local concluída.")
            print("-" * 30)

            print(f"🧠 {agent_state.agent_name} enviando resultado da execução de volta para o Core (stream)...")
            agent_state.add_to_history("user", execution_feedback_for_ai) 

            response_after_execution_stream = model_chat.generate_content(
                agent_state.conversation_history, stream=True
            )
            
            print(f"{agent_state.agent_name} (após execução do código):")
            ai_post_execution_text_accumulated = "" # Redefinir para esta nova resposta
            ai_post_execution_text_accumulated = process_response_stream(response_after_execution_stream, "ai_post_execution_text_accumulated", agent_state.agent_name)

            if not ai_post_execution_text_accumulated.strip():
                err_msg_exec = f"[{agent_state.agent_name} WARN] Resposta pós-execução do LLM vazia após stream."
                print(err_msg_exec) # Já impresso pela função auxiliar
                agent_state.add_to_history("model", err_msg_exec)
            else:
                agent_state.add_to_history("model", ai_post_execution_text_accumulated)
            print("-" * 30)
        
        agent_state.save_state()

        if interaction_count % REFLECTION_INTERVAL == 0 and interaction_count > 0:
            print(f"\nℹ️ {interaction_count} interações. {agent_state.agent_name} agendando ciclo de 'sonho' e evolução...")
            perform_reflection_cycle(agent_state, model_chat)
        
        if interaction_count % GOAL_ACTION_INTERVAL == 0 and interaction_count > 0 and agent_state.active_goals:
             print(f"\nℹ️ {interaction_count} interações. {agent_state.agent_name} considerando agir sobre uma meta...")
             attempt_goal_action(agent_state, model_chat)

    except KeyboardInterrupt:
        print(f"\n🚫 Interrupção pelo usuário. {agent_state.agent_name} encerrando..."); agent_state.save_state(); break
    except google_exceptions.ResourceExhausted:
        print(f"🔴 ERRO CRÍTICO: Atingido o limite de taxa da API do Gemini. {agent_state.agent_name} encerrando."); agent_state.save_state(); break
    except Exception as e:
        print(f"❌ Ocorreu um erro CRÍTICO no loop principal de {agent_state.agent_name}: {e}")
        import traceback; traceback.print_exc(); agent_state.save_state()

# ---------------
# Fim do Script
# ---------------
print(f"\n{agent_state.agent_name} shutdown.")
