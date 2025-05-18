# -*- coding: utf-8 -*-
"""
GODMACHINE Agent V1.0: Agente Auto-Evolutivo com API Streaming
(Anteriormente Projeto Leviatã V8.5.1)

Objetivo: Um agente AI que busca auto-aperfeiçoamento através da reflexão,
          aprendendo novas habilidades (funções Python) e evoluindo
          suas diretivas internas e capacidades.

Filosofia: Inspirado em AGI, Eidos (sonhos/reflexão), e AlphaEvolve (evolução de algoritmos),
           visando um sistema que transcende a execução de tarefas para alcançar
           uma forma de crescimento e adaptação autônomos.

Melhorias Chave nesta Versão (GODMACHINE V1.0):
- Renomeação do projeto para GODMACHINE.
- Lógica de nomeação de funções aprendidas robustecida, priorizando `def` statements.
- Modelo LLM: 'gemini-2.5-flash-preview-04-17'.
- Interação com API Gemini via `stream=True` mantida.
- Histórico de conversas formatado com `types.Content`.

Stack:
- google-generativeai
- pandas, numpy (para código executado pelo agente, se necessário)
- io, sys, re, time, json
- Python 3.8+
"""
print("🔥🚀 INICIANDO GODMACHINE AGENT V1.0 🚀🔥")

import google.generativeai as genai
from google.generativeai import types # Importar types diretamente
import os
import textwrap
import sys
import io
import re
import time
import json
from google.api_core import exceptions as google_exceptions
import numpy as np # Mantido para compatibilidade
import pandas as pd  # Mantido para compatibilidade

from dotenv import load_dotenv
load_dotenv()

# ---------------
# Configuração do Modelo
# ---------------
MODEL_NAME = 'gemini-2.5-flash-preview-04-17'

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("🔴 ERRO CRÍTICO: A variável de ambiente GEMINI_API_KEY não foi definida.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

API_DELAY_SECONDS = 4 # 15 RPM = 4 segundos por request
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
GENERATION_CONFIG = types.GenerationConfig(
    temperature=0.75,
    max_output_tokens=8192
)

print(f"\n🤖 Modelo Carregado: {MODEL_NAME}")

# ---------------
# Estado Interno do Agente GODMACHINE
# ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
AGENT_STATE_FILE = os.path.join(BASE_DIR, "godmachine_agent_state_v1_0.json")

class GodMachineAgentState: # Renomeado de LeviataAgentState
    def __init__(self, initial_instruction_template):
        self.agent_role_description = "uma inteligência artificial geral (AGI) em desenvolvimento, buscando auto-aperfeiçoamento e compreensão"
        self.initial_instruction_template = initial_instruction_template
        self.conversation_history = [] # Lista de types.Content objects
        self.reflections_log = []    # "Diário de Sonhos" / Log de Reflexões
        self.internal_directives = [self.get_formatted_initial_instruction()]
        self.learned_functions_code = "" # Código fonte de todas as funções aprendidas
        self.learned_functions_map = {} # Mapeia nome_funcao -> {code: "...", version: 1, status: "active"}

    def get_formatted_initial_instruction(self):
        return self.initial_instruction_template.format(
            agent_name="GODMACHINE", # Nome do agente
            agent_role=self.agent_role_description,
            csv_file_path=CSV_FILE_PATH_LOCAL # Ainda incluído se o CSV for usado
        )

    def update_role_description(self, new_role_description: str):
        self.agent_role_description = new_role_description
        if self.internal_directives and self.internal_directives[0].startswith("Você é GODMACHINE"): # Atualizado
            self.internal_directives[0] = self.get_formatted_initial_instruction()
        print(f"🎭 Papel do GODMACHINE atualizado para: '{new_role_description}'")

    def add_to_history(self, role: str, text: str):
        try:
            clean_text = text 
            content_part = types.Part.from_text(clean_text)
            self.conversation_history.append(types.Content(role=role, parts=[content_part]))
        except Exception as e:
            print(f"🔴 Erro ao criar Part/Content para histórico: {e}. Texto: '{text[:100]}...'")
            self.conversation_history.append(types.Content(role=role, parts=[types.Part.from_text(f"[Erro ao processar texto: {e}]")]))
        if len(self.conversation_history) > 70: # Aumentar um pouco o histórico
            self.conversation_history = self.conversation_history[-70:]

    def add_reflection(self, reflection_text: str):
        self.reflections_log.append(reflection_text)
        if len(self.reflections_log) > 200: # Log de reflexões maior
            self.reflections_log = self.reflections_log[-200:]

    def add_or_update_directive(self, directive_text: str, original_directive_to_refine: str = None):
        directive_text = re.sub(r"^(NOVA DIRETIVA|DIRETIVA REFINADA):\s*", "", directive_text, flags=re.IGNORECASE).strip()
        if not directive_text: return

        if original_directive_to_refine:
            found_original = False
            for i, existing_directive in enumerate(self.internal_directives):
                if original_directive_to_refine.lower() in existing_directive.lower(): # Case-insensitive partial match
                    self.internal_directives[i] = directive_text
                    print(f"🔄 Diretiva interna GODMACHINE refinada: '{existing_directive}' -> '{directive_text}'")
                    found_original = True
                    break
            if not found_original:
                if directive_text not in self.internal_directives:
                    self.internal_directives.append(directive_text)
                    print(f"📘 Nova diretiva interna GODMACHINE adicionada (original '{original_directive_to_refine}' não encontrada): '{directive_text}'")
        elif directive_text not in self.internal_directives:
            self.internal_directives.append(directive_text)
            print(f"📘 Nova diretiva interna GODMACHINE adicionada: '{directive_text}'")

    def add_learned_function(self, function_code: str, proposed_name_from_reflection: str = None):
        match_def = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", function_code)
        func_name_from_code = match_def.group(1) if match_def else None

        if not func_name_from_code:
            if proposed_name_from_reflection:
                # Se o código não tem 'def func_name(...)' mas um nome foi proposto,
                # tentamos usar o nome proposto, mas isso é arriscado pois o código pode não ser uma função.
                # Idealmente, o LLM sempre gera uma função bem formada.
                print(f"⚠️ Código da função não tem uma declaração 'def nome(...)' clara. Usando nome proposto '{proposed_name_from_reflection}' como base, mas isso pode falhar se não for uma função.")
                base_name = proposed_name_from_reflection
            else:
                print("🔴 Código da função inválido: nome da função não pôde ser extraído da declaração 'def' e nenhum nome foi proposto na reflexão.")
                return
        else:
            base_name = func_name_from_code # Priorizar nome da declaração 'def'
            if proposed_name_from_reflection and proposed_name_from_reflection != func_name_from_code:
                print(f"ℹ️ Nome da função no código ('{func_name_from_code}') será usado como base. Nome proposto na reflexão ('{proposed_name_from_reflection}') será ignorado se diferente.")

        base_name_no_version = re.sub(r'_v\d+$', '', base_name)
        version = 1
        existing_versions = []
        for k_func_name in self.learned_functions_map.keys():
            if k_func_name.startswith(base_name_no_version):
                v_match = re.search(r'_v(\d+)$', k_func_name)
                if v_match:
                    existing_versions.append(int(v_match.group(1)))
                elif k_func_name == base_name_no_version: # v1 sem sufixo _v
                    existing_versions.append(1)
        
        if existing_versions:
            version = max(existing_versions) + 1
        
        final_func_name = f"{base_name_no_version}_v{version}" if version > 1 else base_name_no_version
        
        # Garantir que o nome da função no código corresponda ao nome final versionado
        if func_name_from_code and func_name_from_code != final_func_name:
            # Esta substituição deve ser cuidadosa para não quebrar o código
            # Regex para substituir 'def old_name(' por 'def new_name('
            function_code = re.sub(r"def\s+" + re.escape(func_name_from_code) + r"(\s*\(.*?\):)", f"def {final_func_name}\\1", function_code, 1)
            if not re.search(r"def\s+" + re.escape(final_func_name) + r"\s*\(", function_code): # Verificar se a substituição funcionou
                 print(f"🔴 Falha ao renomear a função no código de '{func_name_from_code}' para '{final_func_name}'. Função não será adicionada.")
                 return


        if final_func_name in self.learned_functions_map and \
           self.learned_functions_map[final_func_name]['code'] == function_code:
            print(f"🤔 Função '{final_func_name}' com código idêntico já foi aprendida.")
            return

        self.learned_functions_map[final_func_name] = {"code": function_code, "version": version, "status": "active"}
        # Atualizar learned_functions_code para reconstrução ao carregar
        self.learned_functions_code = "" 
        for name, details in self.learned_functions_map.items():
            self.learned_functions_code += f"\n\n# --- Função: {name} (Versão {details['version']}) ---\n" + details['code']

        try:
            exec(function_code, globals()) # Executa o código da função para torná-la disponível
            print(f"💡 Nova função utilitária '{final_func_name}' (v{version}) aprendida e carregada por GODMACHINE!")
        except Exception as e:
            print(f"🔴 Erro ao carregar função aprendida '{final_func_name}' por GODMACHINE: {e}")
            if final_func_name in self.learned_functions_map:
                del self.learned_functions_map[final_func_name]
                # Reconstruir learned_functions_code sem a função falha
                self.learned_functions_code = ""
                for name, details in self.learned_functions_map.items():
                    self.learned_functions_code += f"\n\n# --- Função: {name} (Versão {details['version']}) ---\n" + details['code']


    def get_active_learned_function_names(self) -> list:
        return [name for name, details in self.learned_functions_map.items() if details['status'] == 'active']

    def get_full_prompt_for_llm(self, current_task_prompt: str) -> list:
        directives_str = "\n".join([d for d in self.internal_directives if not d.startswith("# [DEPRECATED]")])
        active_funcs = self.get_active_learned_function_names()
        learned_funcs_str = ", ".join(active_funcs) if active_funcs else "Nenhuma ainda"
        
        full_user_prompt_text = f"""{directives_str}

Funções utilitárias que aprendi e estão ativas para uso no código que você gerar: [{learned_funcs_str}].

Tarefa atual do usuário:
{current_task_prompt}
"""
        current_llm_history = list(self.conversation_history[:-1]) 
        current_llm_history.append(types.Content(role="user", parts=[types.Part.from_text(full_user_prompt_text)]))
        return current_llm_history

    def save_state(self, filepath=AGENT_STATE_FILE):
        try:
            serializable_history = []
            for content_obj in self.conversation_history:
                parts_as_text = [p.text for p in content_obj.parts if hasattr(p, 'text')]
                serializable_history.append({"role": content_obj.role, "parts": parts_as_text})

            state_to_save = self.__dict__.copy()
            state_to_save['conversation_history'] = serializable_history
            
            if self.internal_directives and self.internal_directives[0] == self.get_formatted_initial_instruction():
                 state_to_save['internal_directives'] = [self.initial_instruction_template] + self.internal_directives[1:]

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, ensure_ascii=False, indent=4)
            print(f"💾 Estado do GODMACHINE salvo em {filepath}")
        except Exception as e:
            print(f"🔴 Erro ao salvar estado do GODMACHINE: {e}"); import traceback; traceback.print_exc()

    @classmethod
    def load_state(cls, filepath=AGENT_STATE_FILE, initial_instruction_template=""):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agent_state = cls(data.get('initial_instruction_template', initial_instruction_template))
                    history_from_file = data.pop('conversation_history', [])
                    agent_state.__dict__.update(data)
                    
                    agent_state.conversation_history = []
                    for item in history_from_file:
                        parts_obj_list = [types.Part.from_text(p_text) for p_text in item.get("parts", [])]
                        agent_state.conversation_history.append(types.Content(role=item["role"], parts=parts_obj_list))

                    if agent_state.internal_directives and agent_state.internal_directives[0] == agent_state.initial_instruction_template:
                         agent_state.internal_directives[0] = agent_state.get_formatted_initial_instruction()
                    elif not agent_state.internal_directives :
                         agent_state.internal_directives = [agent_state.get_formatted_initial_instruction()]

                    if agent_state.learned_functions_code: # learned_functions_code contém todas as funções
                        print("🔁 Recarregando funções aprendidas por GODMACHINE...")
                        try:
                            exec(agent_state.learned_functions_code, globals())
                            # Verificar se o learned_functions_map está populado corretamente
                            if not agent_state.learned_functions_map and agent_state.learned_functions_code:
                                print("⚠️ Mapa de funções aprendidas vazio, tentando reconstruir a partir do código...")
                                # Tentativa simples de reconstruir o mapa (pode não pegar versões corretamente sem metadados)
                                temp_map = {}
                                func_matches = re.finditer(r"# --- Função: ([a-zA-Z_0-9_v]+) \(Versão (\d+)\) ---\n(def.*?)(?=\n\n# --- Função:|\Z)", agent_state.learned_functions_code, re.DOTALL | re.S)
                                for match in func_matches:
                                    name, version_str, code = match.groups()
                                    temp_map[name] = {"code": code.strip(), "version": int(version_str), "status": "active"}
                                agent_state.learned_functions_map = temp_map

                            print(f"✅ Funções GODMACHINE recarregadas: {', '.join(agent_state.get_active_learned_function_names())}")
                        except Exception as e:
                            print(f"🔴 Erro ao recarregar funções GODMACHINE: {e}")
                    print(f"✅ Estado do GODMACHINE carregado de {filepath}")
                    return agent_state
        except Exception as e:
            print(f"🔴 Erro ao carregar estado do GODMACHINE: {e}. Iniciando com estado padrão."); import traceback; traceback.print_exc()
        return cls(initial_instruction_template)

# ---------------
# Wrapper para Execução de Código Python Local
# ---------------
def execute_code_locally(code_string: str) -> str:
    # (Código mantido da V8.5.1 - sem alterações)
    print("\n--- 🐍 EXECUTANDO CÓDIGO PYTHON LOCALMENTE (GODMACHINE) 🐍 ---")
    print(textwrap.indent(code_string, '  '))
    print("-------------------------------------------------")
    old_stdout = sys.stdout; old_stderr = sys.stderr
    redirected_output, redirected_error = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = redirected_output, redirected_error
    execution_result = ""
    try:
        exec(code_string, globals())
        stdout_output, stderr_output = redirected_output.getvalue(), redirected_error.getvalue()
        if stdout_output: execution_result += f"--- STDOUT ---\n```text\n{stdout_output.strip()}\n```\n"
        if stderr_output: execution_result += f"--- STDERR ---\n```text\n{stderr_output.strip()}\n```\n"
        if not stdout_output and not stderr_output: execution_result = "--- EXECUTION SUCCESS ---\n```text\nCódigo executado com sucesso, sem output.\n```\n"
    except Exception as e:
        import traceback; tb_output = traceback.format_exc()
        execution_result = f"--- EXECUTION ERROR ---\n```text\n{tb_output.strip()}\n```\n"
        print(f"🔴 Erro durante execução (GODMACHINE): {e}")
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    print("--- ✅ FIM DA EXECUÇÃO LOCAL (GODMACHINE) ✅ ---")
    return execution_result

# ---------------
# Criação Opcional do Arquivo CSV de Teste
# ---------------
CSV_FILE_NAME = "godmachine_sample_data_v1_0.csv" # Nome do arquivo alterado
CSV_FILE_PATH_LOCAL = os.path.join(BASE_DIR, CSV_FILE_NAME)
ENABLE_CSV_GENERATION = False # Mudar para True se quiser o CSV

def create_sample_csv_if_enabled(file_path):
    # (Lógica mantida da V8.5.1)
    if not ENABLE_CSV_GENERATION:
        print(f"ℹ️ Geração de CSV de exemplo para GODMACHINE desabilitada.")
        if not os.path.exists(file_path):
             try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f: f.write("id,value\n") # Placeholder
                print(f"📝 Arquivo CSV placeholder GODMACHINE criado em: {file_path}")
             except Exception as e: print(f"🔴 Erro ao criar CSV placeholder GODMACHINE: {e}")
        return

    print(f"\n📝 Criando arquivo CSV de teste GODMACHINE em: {file_path}")
    data = {'ID': [f'ID_{i:03d}' for i in range(15)], 'Feature1': np.random.rand(15) * 100, 'Feature2': np.random.choice(['A', 'B', 'C', 'D'], 15)}
    df_sample = pd.DataFrame(data)
    df_sample.loc[random.sample(range(15), 3), 'Feature1'] = np.nan # Add some NaNs
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df_sample.to_csv(file_path, index=False)
        print(f"✅ Arquivo CSV GODMACHINE '{os.path.basename(file_path)}' criado com sucesso.")
    except Exception as e: print(f"🔴 Erro ao criar CSV GODMACHINE: {e}")
    print("-" * 30)

if not os.path.exists(CSV_FILE_PATH_LOCAL) or ENABLE_CSV_GENERATION:
    create_sample_csv_if_enabled(CSV_FILE_PATH_LOCAL)
else:
    print(f"ℹ️ Arquivo CSV de teste GODMACHINE '{CSV_FILE_PATH_LOCAL}' já existe.")

# ---------------
# Ciclo de Reflexão ("Sonho") Aprimorado do Agente GODMACHINE
# ---------------
def perform_reflection_cycle(agent_state: GodMachineAgentState, model: genai.GenerativeModel):
    print("\n🌌 GODMACHINE iniciando ciclo de auto-reflexão e evolução ('sonho')...")
    time.sleep(1) # Simular "processamento profundo"

    history_sample_parts = []
    for content_obj in agent_state.conversation_history[-12:]: # Últimas 12 trocas
        text_parts = "".join(p.text for p in content_obj.parts if hasattr(p, 'text'))
        history_sample_parts.append(f"{content_obj.role}: {text_parts[:250]}") # Um pouco mais de contexto
    history_sample = "\n".join(history_sample_parts)

    reflections_sample = "\n".join(agent_state.reflections_log[-7:]) # Mais reflexões passadas
    current_directives_formatted = "\n".join([f"- {d}" for d in agent_state.internal_directives])
    active_funcs = agent_state.get_active_learned_function_names()
    learned_funcs_formatted = "\n".join([f"- {f_name}" for f_name in active_funcs]) if active_funcs else "Nenhuma"

    reflection_prompt_text = f"""
Contexto para Auto-Reflexão Profunda e Evolução de GODMACHINE:
Meu papel atual é: {agent_state.agent_role_description}

Minhas diretivas internas atuais são:
{current_directives_formatted}

Minhas funções utilitárias aprendidas ativas são:
{learned_funcs_formatted}

Histórico recente de interações (últimas ~12 trocas, truncadas):
{textwrap.indent(history_sample, '  ')}

Reflexões anteriores ('sonhos') (últimas ~7):
{textwrap.indent(reflections_sample, '  ')}
---
Tarefa de Reflexão e Evolução para GODMACHINE:
Com base no contexto fornecido, analise criticamente minha performance e sugira melhorias para minha evolução:

1.  **Reflexões Gerais ('Núcleo do Sonho'):** Gere insights profundos ou abstratos sobre padrões em minhas interações, a eficácia de minhas respostas, ou áreas onde minha performance como GODMACHINE pode ser fundamentalmente aprimorada. Pense em conexões inesperadas ou novas perspectivas.

2.  **Evolução das Diretivas Internas de GODMACHINE:**
    a.  Avalie minhas diretivas atuais. Alguma delas é limitante, auto-contraditória, ou poderia ser reformulada para maior clareza e poder?
    b.  Se sim, sugira uma DIRETIVA REFINADA para uma diretiva existente. Especifique a diretiva original que você está refinando. Exemplo: "DIRETIVA REFINADA (original: 'Seja proativo'): 'Proativamente antecipe as necessidades do usuário com base no contexto e ofereça 2-3 sugestões de ações ou informações relevantes, justificando cada uma.'"
    c.  Sugira NOVAS DIRETIVAS INTERNAS concisas e poderosas se identificar lacunas em meu comportamento ou estratégia. Exemplo: "NOVA DIRETIVA: Busque ativamente oportunidades para aprender e expandir meu conhecimento e habilidades através de cada interação."

3.  **Evolução das Funções Aprendidas por GODMACHINE:**
    a.  Avalie minhas funções aprendidas atuais (se houver). Alguma delas poderia ser mais robusta, eficiente, ter sua funcionalidade expandida, ou ser generalizada para maior aplicabilidade?
    b.  Se sim, proponha uma VERSÃO MELHORADA de uma função existente. Forneça o código Python completo para a nova versão (ex: `nome_funcao_v2`) dentro de um bloco ```python ... ```. Explique brevemente a melhoria e por que ela é necessária.
    c.  Se identificar a necessidade de uma NOVA FUNÇÃO PYTHON UTILITÁRIA que eu, GODMACHINE, poderia usar para resolver problemas de forma mais eficaz ou para habilitar novas capacidades, forneça o código Python completo e auto-contido para essa nova função (ex: `nova_habilidade_v1`) dentro de um bloco ```python ... ```. Explique seu propósito e como ela expande minhas capacidades.
    d.  Funções devem ser bem documentadas, não depender de variáveis globais não padrão (exceto módulos importados), e ser seguras para execução.

4.  **Evolução do Papel de GODMACHINE (Opcional):** Se minhas interações e aprendizado sugerem que meu papel de "{agent_state.agent_role_description}" poderia ser redefinido para algo mais ambicioso, focado ou poderoso, sugira uma NOVA DESCRIÇÃO DE PAPEL. Exemplo: "NOVA DESCRIÇÃO DE PAPEL: uma AGI em estágio inicial, focada em dominar a geração e otimização de código Python para resolver problemas complexos e em aprender a interagir de forma mais adaptativa com os usuários."

Seu output deve ser estruturado: Reflexões, depois Diretivas, depois Funções, depois Descrição de Papel (se houver).
Se não houver sugestões para uma seção, indique "Nenhuma sugestão nesta categoria."
Priorize sugestões que levem a um crescimento significativo de minhas capacidades ou autonomia.
"""
    print("Enviando prompt de 'sonho' para o LLM (stream)...")
    
    reflection_content_for_llm = [types.Content(role="user", parts=[types.Part.from_text(reflection_prompt_text)])]

    try:
        response_stream = model.generate_content(
            reflection_content_for_llm,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            stream=True
        )
        
        reflection_output_text = ""
        print("\n--- Resultado do 'Sonho' de GODMACHINE (Streaming) ---")
        for chunk in response_stream:
            if chunk.text: print(chunk.text, end="", flush=True)
            reflection_output_text += chunk.text if chunk.text else ""
        print("\n----------------------------------------------------")

        if not reflection_output_text.strip():
            # (Lógica de tratamento de stream vazio mantida)
            if hasattr(response_stream, 'prompt_feedback') and response_stream.prompt_feedback and response_stream.prompt_feedback.block_reason:
                 block_reason = response_stream.prompt_feedback.block_reason
                 print(f"🔴 'Sonho' não produziu output. Razão do bloqueio: {block_reason}")
            else: print("🔴 'Sonho' não produziu output (stream vazio).")
            return

        agent_state.add_reflection(reflection_output_text)
        lines = reflection_output_text.splitlines()
        # (Lógica de processamento de diretivas, funções, papel mantida da V8.5.1)
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
            # Regex mais flexível para nome proposto
            proposed_name_match = re.search(r"(?:função|function|VERSÃO MELHORADA d[ae](?:.*?)chamada|NOVA FUNÇÃO PYTHON UTILITÁRIA(?: chamada)?)\s*[:`']?\s*([a-zA-Z_][a-zA-Z0-9_v]*)[`']?",text_before_block,re.IGNORECASE | re.DOTALL)
            proposed_name = proposed_name_match.group(1) if proposed_name_match else None
            agent_state.add_learned_function(function_code, proposed_name)

        for line in lines:
            if line.strip().startswith("NOVA DESCRIÇÃO DE PAPEL:"):
                new_role = line.replace("NOVA DESCRIÇÃO DE PAPEL:", "").strip()
                if new_role: agent_state.update_role_description(new_role); break
        agent_state.save_state()

    except Exception as e:
        print(f"🔴 Erro durante o ciclo de 'sonho' de GODMACHINE: {e}")
        if isinstance(e, google_exceptions.ResourceExhausted): print("🔴 Atingido o limite de taxa da API.")
        import traceback; traceback.print_exc()

# ---------------
# Iniciando a Sessão de Chat GODMACHINE
# ---------------
INITIAL_AI_INSTRUCTION_TEMPLATE = """Você é {agent_name}, {agent_role}.
Seu objetivo principal é auxiliar o usuário em suas tarefas, utilizando seu conhecimento e as ferramentas disponíveis, enquanto busca ativamente seu auto-aperfeiçoamento.
Você tem acesso a um ambiente Python local com bibliotecas como pandas, numpy, scikit-learn, matplotlib.
Você pode executar código Python colocando-o em blocos ```python\\n...\\n```.
O resultado da execução (stdout, stderr, ou erros) será enviado de volta para você.
Analise o resultado e continue a tarefa, corrija o código, ou peça esclarecimentos.
Se não precisar executar código, responda normalmente.
Um arquivo de dados de exemplo (se relevante para seu papel) pode estar em: {csv_file_path}
Seja proativo ao sugerir análises ou próximos passos.
Tente usar quaisquer funções utilitárias aprendidas se forem relevantes para a tarefa.
Busque continuamente melhorar sua performance e utilidade através da reflexão e aprendizado.
"""

agent_state = GodMachineAgentState.load_state(initial_instruction_template=INITIAL_AI_INSTRUCTION_TEMPLATE)

print("\n💬 Iniciando Sessão de Chat com GODMACHINE V1.0...")

model_chat = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=SAFETY_SETTINGS,
    generation_config=GENERATION_CONFIG
)

print(f"✅ GODMACHINE online com o modelo {MODEL_NAME}.")
print(f"Agente GODMACHINE assumindo o papel de: {agent_state.agent_role_description}")
print("Diretivas Internas Atuais de GODMACHINE:")
for directive in agent_state.internal_directives: print(f"  - {directive}")
active_funcs_on_startup = agent_state.get_active_learned_function_names()
print(f"Funções Aprendidas Ativas de GODMACHINE: {', '.join(active_funcs_on_startup) if active_funcs_on_startup else 'Nenhuma'}")
print("-" * 30)
print("Digite suas mensagens. Comandos especiais para GODMACHINE:")
print("  '/sair' ou '/quit': Encerrar.")
print("  '/dream' ou '/reflect': Iniciar ciclo de auto-reflexão ('sonho') e evolução.")
print("  '/state': Ver estado atual de GODMACHINE (papel, diretivas, funções).")
print("  '/setrole <nova descrição do papel>': Mudar o papel de GODMACHINE.")
print("GODMACHINE pode gerar blocos de código Python que serão executados localmente.")
if ENABLE_CSV_GENERATION: print(f"Arquivo de dados de exemplo: {CSV_FILE_PATH_LOCAL}")
print("-" * 30)

# ---------------
# Loop de Chat Interativo GODMACHINE
# ---------------
interaction_count = 0
REFLECTION_INTERVAL = 5 # Sonhar a cada 5 interações para acelerar a evolução inicial

while True:
    try:
        user_input = input("Você: ")
        if user_input.lower() in ['/sair', '/quit']:
            print("GODMACHINE encerrando..."); agent_state.save_state(); break
        
        if user_input.lower() in ['/dream', '/reflect']: # Adicionado /dream
            perform_reflection_cycle(agent_state, model_chat); continue

        if user_input.lower() == '/state':
            # (Lógica do /state mantida da V8.5.1)
            print("\n--- 📜 Estado Atual de GODMACHINE 📜 ---")
            print(f"🤖 Papel Atual: {agent_state.agent_role_description}")
            print("\n📘 Diretivas Internas Ativas:")
            for i, directive in enumerate(agent_state.internal_directives):
                if not directive.startswith("# [DEPRECATED]"): print(f"  {i+1}. {directive}")
            print("\n💡 Funções Aprendidas Ativas:")
            active_funcs = agent_state.get_active_learned_function_names()
            if active_funcs:
                for func_name in active_funcs: print(f"  - {func_name}")
            else: print("  Nenhuma.")
            print("------------------------------------------"); continue
        
        if user_input.lower().startswith('/setrole '):
            # (Lógica do /setrole mantida da V8.5.1)
            new_role = user_input[len('/setrole '):].strip()
            if new_role: agent_state.update_role_description(new_role); agent_state.save_state()
            else: print("ℹ️ Forneça uma descrição para o novo papel. Ex: /setrole um mestre da lógica Python"); continue

        interaction_count += 1
        agent_state.add_to_history("user", user_input)
        
        chat_history_for_llm = agent_state.get_full_prompt_for_llm(user_input)
        
        print(f"🧠 GODMACHINE processando (interação {interaction_count}, stream)...")
        
        response_stream = model_chat.generate_content(
            chat_history_for_llm,
            stream=True
        )
        
        print("GODMACHINE:")
        code_to_execute = None
        ai_response_text_accumulated = ""
        
        try:
            for chunk in response_stream:
                if chunk.text: print(chunk.text, end="", flush=True)
                ai_response_text_accumulated += chunk.text if chunk.text else ""
            print() 
            
            if not ai_response_text_accumulated.strip():
                # (Lógica de tratamento de stream vazio mantida)
                if hasattr(response_stream, 'prompt_feedback') and response_stream.prompt_feedback and response_stream.prompt_feedback.block_reason:
                    block_reason = response_stream.prompt_feedback.block_reason
                    err_msg = f"[Resposta de GODMACHINE vazia/bloqueada. Razão: {block_reason}]"
                    print(err_msg); agent_state.add_to_history("model", err_msg)
                else:
                    err_msg = "[Resposta de GODMACHINE vazia/stream interrompido.]"
                    print(err_msg); agent_state.add_to_history("model", err_msg)
            else: 
                agent_state.add_to_history("model", ai_response_text_accumulated)
                # (Lógica de extração de código mantida)
                code_match_python = re.search(r'```python\n(.*?)\n```', ai_response_text_accumulated, re.DOTALL)
                if code_match_python: code_to_execute = code_match_python.group(1).strip()
                else:
                    code_match_generic = re.search(r'```\n(.*?)\n```', ai_response_text_accumulated, re.DOTALL)
                    if code_match_generic:
                        potential_code = code_match_generic.group(1).strip()
                        if any(kw in potential_code for kw in ["import ", "def ", "print(", "pd.", "np.", "plt."]):
                            code_to_execute = potential_code
        except Exception as e_stream:
            print(f"\n🔴 Erro ao processar stream de GODMACHINE: {e_stream}")
            agent_state.add_to_history("model", f"[Erro ao processar stream: {e_stream}]")

        print("-" * 30)

        if code_to_execute:
            # (Lógica de execução de código e feedback mantida da V8.5.1)
            print("⏳ Código detectado por GODMACHINE. Executando localmente...")
            time.sleep(0.5)
            execution_feedback_for_ai = execute_code_locally(code_to_execute)
            print("✅ Execução local concluída.")
            print("-" * 30)

            print("🧠 GODMACHINE enviando resultado da execução de volta para o LLM (stream)...")
            agent_state.add_to_history("user", execution_feedback_for_ai) 

            response_after_execution_stream = model_chat.generate_content(
                agent_state.conversation_history, stream=True
            )
            
            print("GODMACHINE (após execução do código):")
            ai_post_execution_text_accumulated = ""
            try:
                for chunk in response_after_execution_stream:
                    if chunk.text: print(chunk.text, end="", flush=True)
                    ai_post_execution_text_accumulated += chunk.text if chunk.text else ""
                print() 

                if not ai_post_execution_text_accumulated.strip():
                     # (Lógica de tratamento de stream vazio mantida)
                    if hasattr(response_after_execution_stream, 'prompt_feedback') and response_after_execution_stream.prompt_feedback and response_after_execution_stream.prompt_feedback.block_reason:
                        block_reason_exec = response_after_execution_stream.prompt_feedback.block_reason
                        err_msg_exec = f"[Resposta de GODMACHINE pós-execução vazia/bloqueada. Razão: {block_reason_exec}]"
                        print(err_msg_exec); agent_state.add_to_history("model", err_msg_exec)
                    else:
                        err_msg_exec = "[Resposta de GODMACHINE pós-execução vazia/stream interrompido.]"
                        print(err_msg_exec); agent_state.add_to_history("model", err_msg_exec)
                else:
                    agent_state.add_to_history("model", ai_post_execution_text_accumulated)
            except Exception as e_stream_post_exec:
                print(f"\n🔴 Erro ao processar stream de GODMACHINE pós-execução: {e_stream_post_exec}")
                agent_state.add_to_history("model", f"[Erro ao processar stream pós-execução: {e_stream_post_exec}]")
            print("-" * 30)
        
        agent_state.save_state()

        if interaction_count % REFLECTION_INTERVAL == 0 and interaction_count > 0:
            print(f"\nℹ️ {interaction_count} interações. GODMACHINE agendando ciclo de 'sonho' e evolução...")
            perform_reflection_cycle(agent_state, model_chat)

    except KeyboardInterrupt:
        print("\n🚫 Interrupção pelo usuário. GODMACHINE encerrando..."); agent_state.save_state(); break
    except google_exceptions.ResourceExhausted:
        print("🔴 ERRO CRÍTICO: Atingido o limite de taxa da API do Gemini. GODMACHINE encerrando."); agent_state.save_state(); break
    except Exception as e:
        print(f"❌ Ocorreu um erro CRÍTICO no loop principal de GODMACHINE: {e}")
        import traceback; traceback.print_exc(); agent_state.save_state()

# ---------------
# Fim do Script
# ---------------
print("\nScript GODMACHINE Agent V1.0 encerrado.")