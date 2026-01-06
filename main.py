import os
import time
import threading
from typing import TypedDict, Optional
from flask import Flask
from dotenv import load_dotenv

# LangChain / LangGraph / Supabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from supabase import create_client, Client

# Carrega variáveis locais se existirem (para testes no seu PC)
load_dotenv()

# ==========================================
# PARTE 1: CONFIGURAÇÃO DO SERVIDOR (O "HACK")
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "O Agente DeepSeek está rodando em background! 🚀"

def run_flask():
    # O Render injeta a variável PORT automaticamente
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ==========================================
# PARTE 2: CONFIGURAÇÃO DO AGENTE (SEU CÓDIGO)
# ==========================================

# Configuração DeepSeek (Via OpenAI Client)
llm = ChatOpenAI(
    model='deepseek-chat', 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url='https://api.deepseek.com',
    temperature=0.1
)

# Inicializa Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltam as variáveis do Supabase!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- HELPERS ---
def get_agent_config(agent_name):
    try:
        response = supabase.table("agent_configs").select("*").eq("agent_name", agent_name).execute()
        if response.data:
            return response.data[0]
        return {"system_instruction": "Você é um assistente útil.", "temperature": 0.1}
    except Exception as e:
        print(f"[ERRO CONFIG] {e}")
        return {"system_instruction": "Erro.", "temperature": 0.1}

def log_execution(agent_name, level, message):
    print(f"[{level}] {agent_name}: {message}")
    try:
        supabase.table("execution_logs").insert({
            "agent_name": agent_name, "log_level": level, "message": str(message)
        }).execute()
    except Exception:
        pass

# --- ESTADO E NÓS ---
class ContentState(TypedDict):
    row_id: Optional[str]
    raw_text: Optional[str]
    source_type: Optional[str]
    key_insights: Optional[str]
    final_draft: Optional[str]
    status: str

def ingest_node(state: ContentState):
    print("--- 1. INGESTOR ---")
    response = supabase.table("raw_materials").select("*").eq("status", "pending").limit(1).execute()
    data = response.data
    
    if not data:
        return {"status": "no_data"}
    
    item = data[0]
    # Trava o item
    supabase.table("raw_materials").update({"status": "processing"}).eq("id", item['id']).execute()
        
    return {
        "row_id": item['id'],
        "raw_text": item['content_text'],
        "source_type": item['source_type'],
        "status": "has_data"
    }

def curator_node(state: ContentState):
    agent_name = "Curator"
    config = get_agent_config(agent_name)
    msg = [SystemMessage(content=config['system_instruction']), HumanMessage(content=state['raw_text'])]
    try:
        response = llm.invoke(msg)
        return {"key_insights": response.content}
    except Exception as e:
        log_execution(agent_name, "ERROR", str(e))
        return {"status": "failed"}

def writer_node(state: ContentState):
    agent_name = "Ghostwriter"
    config = get_agent_config(agent_name)
    prompt = f"{config['system_instruction']}\n\nINSIGHTS:\n{state['key_insights']}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_draft": response.content}
    except Exception as e:
        log_execution(agent_name, "ERROR", str(e))
        return {"status": "failed"}

def publisher_node(state: ContentState):
    print("--- 4. PUBLISHER ---")
    supabase.table("ready_materials").insert({
        "raw_material_id": state['row_id'],
        "platform": "general",
        "final_content": state['final_draft'],
        "virality_score": 85
    }).execute()
    
    supabase.table("raw_materials").update({"status": "done"}).eq("id", state['row_id']).execute()
    return {"status": "finished"}

# --- GRAFO ---
builder = StateGraph(ContentState)
builder.add_node("Ingestor", ingest_node)
builder.add_node("Curator", curator_node)
builder.add_node("Writer", writer_node)
builder.add_node("Publisher", publisher_node)
builder.add_edge(START, "Ingestor")

def route_ingestor(state: ContentState):
    if state["status"] == "no_data": return END
    return "Curator"

builder.add_conditional_edges("Ingestor", route_ingestor)
builder.add_edge("Curator", "Writer")
builder.add_edge("Writer", "Publisher")
builder.add_edge("Publisher", END)

content_engine = builder.compile()

# ==========================================
# PARTE 3: LOOP INFINITO (WORKER)
# ==========================================
def run_agent_worker():
    print(">> WORKER INICIADO: Usina de Conteúdo DeepSeek <<")
    while True:
        try:
            inputs = {"status": "check"} 
            result = content_engine.invoke(inputs)
            
            if "Publisher" in result:
                 print(f"[SUCESSO] Conteúdo processado.")
            else:
                print("[INFO] Nada para processar. Dormindo...")
            
            time.sleep(10) # Pausa entre ciclos
            
        except Exception as e:
            print(f"[ERRO CRÍTICO] {e}")
            time.sleep(10)

# ==========================================
# PARTE 4: EXECUÇÃO HÍBRIDA
# ==========================================
if __name__ == "__main__":
    # 1. Inicia o Worker em uma Thread Separada (Background)
    worker_thread = threading.Thread(target=run_agent_worker)
    worker_thread.daemon = True # Morre se o processo principal morrer
    worker_thread.start()

    # 2. Inicia o Web Server (Bloqueia o processo principal e mantém o Render feliz)
    run_flask()