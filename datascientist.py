# -*- coding: utf-8 -*-
"""
DataScientist Bot — this could be a real product. gemini-powered, python agent, feedback loop. give it a terminal and notebook interface, maybe as a vscode extension or jupyter plugin. if it’s really strong, spin it as a lab assistant, scientist co-pilot.
# Projeto Leviatã V8.2: AI com Execução Local

**Objetivo:** Adaptar o chat interativo com execução de código local (V8.1) para rodar como um script Python independente, fora do ambiente Google Colab.

**Filosofia:** Entregar a capacidade do AI Data Scientist auto-corretivo em um pacote simples e compartilhável (um arquivo .py).

**Componentes:**
- Criação de uma Sessão de Chat (SEM ferramentas built-in)
- Wrapper para Execução de Código Python Local Segura
- Loop de Chat Interativo
- **AJUSTE:** Leitura de API Key apenas de variável de ambiente.
- **AJUSTE:** Criação de arquivo CSV de teste no diretório local do script.
- **REMOÇÃO:** Código específico do Colab.

**Stack:**
- google-generativeai
- pandas, numpy, scikit-learn, matplotlib
- io, sys, re, time
- **REQUISITO:** Python 3.8+ (para re.search no bloco de código)
"""
print("🚀 Iniciando ...")
import google.generativeai as genai
import os
import textwrap
import sys
import io
import re 
import time
from google.genai import types
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------
# Configuração do Modelo
# ---------------

# Modelo que suporta chat e é bom em gerar/entender código
MODEL_NAME = 'gemini-2.5-flash-preview-04-17' 
# MODEL_NAME = 'gemini-1.5-flash-latest'

API_DELAY_SECONDS = 4 # Delay entre chamadas p/ rate limit

print(f"\n🤖 Modelo Carregado: {MODEL_NAME}")

# NÃO CONFIGURAMOS FERRAMENTAS NA INICIALIZAÇÃO PARA EVITAR O ERRO!
# O AI VAI APENAS GERAR O CÓDIGO COMO TEXTO E NOSSO SCRIPT VAI DETECTAR E EXECUTAR.
tools_list = None


# ---------------
# Wrapper para Execução de Código Python Local
# ---------------

def execute_code_locally(code_string: str) -> str:
    """
    Executa um string de código Python localmente e captura stdout/stderr/erros.
    Retorna uma string formatada com o resultado para o AI.
    """
    print("\n--- EXECUTANDO CÓDIGO LOCALMENTE ---")
    print(code_string) # Mostra o código que está sendo executado
    print("------------------------------------")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    execution_result = ""
    success = False
    try:
        # Use exec() para múltiplos statements. globals() para acesso ao ambiente.
        # O ambiente onde o script .py roda é o "local".
        exec(code_string, globals()) # globals() permite acesso a variáveis definidas antes (ex: pandas, numpy, variáveis criadas em execuções anteriores)

        stdout_output = redirected_output.getvalue()
        stderr_output = redirected_error.getvalue()

        if stdout_output:
            execution_result += f"--- STDOUT ---\n```text\n{stdout_output.strip()}\n```\n"
        if stderr_output:
             execution_result += f"--- STDERR ---\n```text\n{stderr_output.strip()}\n```\n"

        if not stdout_output and not stderr_output:
             execution_result = "--- EXECUTION SUCCESS ---\n```text\nCódigo executado com sucesso, sem output.\n```\n"


    except Exception as e:
        import traceback
        tb_output = traceback.format_exc()
        execution_result = f"--- EXECUTION ERROR ---\n```text\n{tb_output.strip()}\n```\n"

    finally:
        # Restaura stdout e stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    print("--- FIM DA EXECUÇÃO LOCAL ---")

    return execution_result


# ---------------
# OPCIONAL
# Criação do Arquivo CSV de Teste Local (no diretório do script)
# ---------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CSV_FILE_NAME = "sample_data.csv"
CSV_FILE_PATH_LOCAL = os.path.join(BASE_DIR, CSV_FILE_NAME)


def create_sample_csv(file_path):
    """Cria um arquivo CSV de exemplo no caminho especificado."""
    print(f"\n📝 Criando arquivo CSV de teste em: {file_path}")
    # Simula um dataset B2B simples com alguns NaNs e tipos misturados
    data = {
        'ClienteID': [f'CLI_{i:03d}' for i in range(10)],
        'ValorServico': [150.5, 220.0, np.nan, 180.0, 300.0, 120.0, 450.0, np.nan, 200.0, 250.0],
        'MesesContrato': [12, 24, 18, 12, 36, 12, 24, 18, 12, 24],
        'TipoCliente': ['Novo', 'Existente', 'Novo', 'Existente', 'Existente', 'Novo', 'Existente', 'Novo', 'Existente', 'Novo'],
        'Ativo': [True, False, True, True, False, True, True, False, True, True]
    }
    df_sample = pd.DataFrame(data)

    df_sample.loc[2, 'TipoCliente'] = np.nan

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df_sample.to_csv(file_path, index=False)
    print(f"✅ Arquivo CSV '{file_path}' criado com sucesso.")
    print("Head do arquivo criado:")
    print(df_sample.head())
    print("-" * 30)


create_sample_csv(CSV_FILE_PATH_LOCAL)


# ---------------
# Iniciando a Sessão de Chat
# ---------------

# Instrução inicial
INITIAL_AI_INSTRUCTION = f"Você é um AI Data Scientist com acesso a um ambiente Python local com as bibliotecas pandas, numpy, scikit-learn e matplotlib. Você pode executar código Python colocando-o em blocos ```python\\n...\\n```. Se o código executar e tiver output ou erro, o resultado será enviado de volta para você para análise. Analise o resultado e continue a tarefa ou corrija o código. Se não precisar executar código, responda normalmente."

print("\n💬 Iniciando Sessão de Chat com Execução Local...")

# Cria a sessão de chat SEM FERRAMENTAS built-in, mas com instrução inicial
model_chat = genai.GenerativeModel(model_name=MODEL_NAME)
chat_session = model_chat.start_chat()
response = chat_session.send_message(INITIAL_AI_INSTRUCTION)

print(f"✅ Sessão de chat iniciada com o modelo {MODEL_NAME}.")
print("AI (Instrução Inicial):")
print(INITIAL_AI_INSTRUCTION)
print("-" * 30)
print("Digite suas mensagens. Digite 'sair' ou 'quit' para encerrar.")
print("O AI pode gerar blocos de código Python que serão executados localmente.")
print("-" * 30)


# ---------------
# Loop de Chat Interativo com Feedback Loop de Execução Local
# ---------------

while True:
    try:
        user_input = input("Você: ")
        if user_input.lower() in ['sair', 'quit']:
            print("Encerrando chat.")
            break

        # --- 7a: Envia a mensagem do usuário e pega a primeira resposta do AI ---

        response = chat_session.send_message(user_input)

        # --- 7b: Processa a resposta do AI ---
        print("AI:")
        code_to_execute = None
        ai_response_text = "" # Acumula todo o texto da resposta do AI neste turno

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            ai_response_text += part.text
                            # Detecta bloco de código Python ou genérico
                            code_match_python = re.search(r'```python\n(.*?)\n```', part.text, re.DOTALL)
                            if code_match_python:
                                code_to_execute = code_match_python.group(1).strip()
                            if code_to_execute is None: # Tenta o genérico se o python não achou
                                code_match_generic = re.search(r'```\n(.*?)\n```', part.text, re.DOTALL)
                                if code_match_generic:
                                    code_to_execute = code_match_generic.group(1).strip()


                    # Imprime a resposta neste turno
                    print(ai_response_text, end="")


                else:
                    # Caso o candidate não tenha content/parts (ex: blocked)
                    if response.finish_reason:
                        print(f"[Resposta finalizada por: {response.finish_reason}]", end="")
                    if response.safety_ratings:
                        print(f"[Safety Ratings: {response.safety_ratings}]", end="")
            # Nova linha após a resposta inicial
            print("\n" + "-" * 30)

        else:
            # Caso a resposta inteira seja blocked ou vazia
            if response.prompt_feedback:
                if response.prompt_feedback.block_reason:
                    print(f"[Resposta BLOQUEADA. Razão: {response.prompt_feedback.block_reason}]")
                if response.prompt_feedback.safety_ratings:
                    print(f"[Prompt Safety Ratings: {response.prompt_feedback.safety_ratings}]")
            else:
                print("[Resposta inicial vazia ou com problema.]")
            print("-" * 30)


        # --- 7c: Executa o código localmente se detectado ---
        if code_to_execute:
            # Adiciona um pequeno delay antes de executar código pra simular "pensamento" ou rate limit
            time.sleep(2)
            execution_feedback_for_ai = execute_code_locally(code_to_execute)
            print("✅ Execução concluída.")
            print("-" * 30)

            # --- 7d: Envia o resultado da execução de volta ---
            # Feedback loop
            print("🧠 Enviando resultado da execução de volta para o AI para o próximo turno...")
            # Formata o resultado como uma mensagem do "ambiente" ou "tool" para o AI
            # Usando um formato de texto simples que o AI entenda que é um resultado de código
            feedback_message_content = f"""
            O código Python que você gerou foi executado.
            Aqui está o resultado da execução:
            {execution_feedback_for_ai}

            Por favor, analise este resultado e continue a conversa com o usuário.
            **Se o código rodou com sucesso e produziu output, por favor, inclua o output relevante (o texto dentro dos blocos ```text\n...\n```) na sua resposta para o usuário.**
            Se houve um erro, explique o erro para o usuário e sugira o próximo passo (talvez um código corrigido), **incluindo o texto do erro na sua resposta.**
            """

            # Adiciona um pequeno delay antes de mandar o resultado de volta (rate limit)
            time.sleep(API_DELAY_SECONDS)
            # Envia a mensagem de feedback
            response_after_execution = chat_session.send_message(feedback_message_content)

            # Processa e imprime a RESPOSTA APÓS VER O RESULTADO
            print("AI (após execução do código):")
            if response_after_execution.candidates:
                for candidate in response_after_execution.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                print(part.text, end="")
                            # Se gerar MAIS código aqui, o loop principal vai detectar na PRÓXIMA rodada
                    else:
                        if response.finish_reason:
                            print(f"[Resposta finalizada por: {response.finish_reason}]", end="")
                        if candidate.safety_ratings:
                            print(f"[Safety Ratings: {candidate.safety_ratings}]", end="")
                print("\n" + "-" * 30) # Nova linha após a resposta completa
            else:
                if response_after_execution.prompt_feedback:
                    if response_after_execution.prompt_feedback.block_reason:
                        print(f"[Resposta BLOQUEADA após execução. Razão: {response.prompt_feedback.block_reason}]")
                    if response_after_execution.prompt_feedback.safety_ratings:
                        print(f"[Prompt Safety Ratings após execução: {response.prompt_feedback.safety_ratings}]")
                    else:
                        print("[Resposta do AI após execução vazia ou com problema.]")
                    print("-" * 30)


        # O loop continua, pedindo nova entrada do usuário ou processando nova resposta
        # (se gerou código e respondeu ao resultado)

    except Exception as e:
        print(f"❌ Ocorreu um erro CRÍTICO no loop principal do chat: {e}")
        # Quebra o loop se der um erro que impeça a continuação
        import traceback
        print(traceback.format_exc())
        break


# ---------------
# 8. Fim do Script
# ---------------
print("\nScript encerrado.")