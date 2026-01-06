import os
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import TypedDict, Optional
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain / LangGraph / Supabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from supabase import create_client, Client

# Carrega variáveis locais se existirem
load_dotenv()

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
shutdown_event = threading.Event()
worker_thread = None

# ==========================================
# PARTE 1: SERVIDOR FLASK (API)
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "operational",
        "service": "DeepSeek Content Engine",
        "version": "1.0.0",
        "worker_active": worker_thread.is_alive() if worker_thread else False
    })

@app.route('/health')
def health():
    """Health check para monitoramento"""
    try:
        # Testa conexão com Supabase
        supabase.table("raw_materials").select("id").limit(1).execute()
        
        return jsonify({
            "status": "healthy",
            "worker_running": worker_thread.is_alive() if worker_thread else False,
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 503

@app.route('/ingest', methods=['POST'])
def ingest_data():
    """
    Recebe um JSON e salva na tabela raw_materials.
    Payload esperado: {"text": "conteúdo aqui", "source": "youtube/twitter/etc"}
    """
    try:
        # Validação de dados
        if not request.json:
            return jsonify({"error": "JSON inválido ou ausente"}), 400
        
        data = request.json
        content_text = data.get('text', '').strip()
        source_type = data.get('source', 'api_push')
        
        # Validações
        if not content_text:
            return jsonify({"error": "Campo 'text' é obrigatório e não pode estar vazio"}), 400
        
        if len(content_text) > 50000:
            return jsonify({"error": "Texto muito grande (max 50.000 caracteres)"}), 413
        
        if len(content_text) < 10:
            return jsonify({"error": "Texto muito curto (min 10 caracteres)"}), 400
        
        # Inserção no Supabase
        response = supabase.table("raw_materials").insert({
            "content_text": content_text,
            "source_type": source_type,
            "status": "pending"
        }).execute()
        
        return jsonify({
            "status": "success",
            "message": "Conteúdo recebido e será processado em breve",
            "id": response.data[0]['id']
        }), 201
        
    except Exception as e:
        log_execution("API", "ERROR", f"Erro no /ingest: {str(e)}")
        return jsonify({"error": "Erro interno ao processar requisição"}), 500

@app.route('/stats')
def stats():
    """Estatísticas do sistema"""
    try:
        pending = supabase.table("raw_materials").select("id", count="exact").eq("status", "pending").execute()
        processing = supabase.table("raw_materials").select("id", count="exact").eq("status", "processing").execute()
        done = supabase.table("raw_materials").select("id", count="exact").eq("status", "done").execute()
        
        return jsonify({
            "queue": {
                "pending": pending.count,
                "processing": processing.count,
                "completed": done.count
            },
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    """Inicia o servidor Flask"""
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# ==========================================
# PARTE 2: CONFIGURAÇÃO DO AGENTE
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
    raise ValueError("❌ Faltam as variáveis SUPABASE_URL e SUPABASE_KEY!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- HELPERS ---
def get_agent_config(agent_name):
    """Busca configuração do agente no Supabase"""
    try:
        response = supabase.table("agent_configs").select("*").eq("agent_name", agent_name).execute()
        if response.data:
            return response.data[0]
        
        # Fallback configs
        default_configs = {
            "Curator": {
                "system_instruction": "Você é um curador de conteúdo. Extraia os insights mais importantes e relevantes do texto fornecido.",
                "temperature": 0.1
            },
            "Ghostwriter": {
                "system_instruction": "Você é um ghostwriter profissional. Crie conteúdo engajante e viral baseado nos insights fornecidos.",
                "temperature": 0.7
            }
        }
        return default_configs.get(agent_name, {"system_instruction": "Você é um assistente útil.", "temperature": 0.1})
    except Exception as e:
        log_execution("CONFIG", "ERROR", f"Erro ao buscar config de {agent_name}: {e}")
        return {"system_instruction": "Você é um assistente útil.", "temperature": 0.1}

def log_execution(agent_name, level, message):
    """Registra logs no console e Supabase"""
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] [{level}] {agent_name}: {message}")
    
    try:
        supabase.table("execution_logs").insert({
            "agent_name": agent_name,
            "log_level": level,
            "message": str(message)[:1000]  # Limita tamanho da mensagem
        }).execute()
    except Exception as e:
        print(f"[ERRO LOG] Falha ao salvar log: {e}")

# --- ESTADO ---
class ContentState(TypedDict):
    row_id: Optional[str]
    raw_text: Optional[str]
    source_type: Optional[str]
    key_insights: Optional[str]
    final_draft: Optional[str]
    status: str
    error_count: int

# --- NÓS DO GRAFO ---
def ingest_node(state: ContentState):
    """Busca o próximo item pendente para processar"""
    log_execution("Ingestor", "INFO", "Verificando fila...")
    
    try:
        # Busca items pendentes OU travados há mais de 10 minutos
        ten_minutes_ago = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        
        response = supabase.table("raw_materials")\
            .select("*")\
            .or_(f"status.eq.pending,and(status.eq.processing,updated_at.lt.{ten_minutes_ago})")\
            .order("created_at", desc=False)\
            .limit(1)\
            .execute()
        
        data = response.data
        
        if not data:
            return {"status": "no_data"}
        
        item = data[0]
        
        # Atualiza status para processing
        supabase.table("raw_materials").update({
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", item['id']).execute()
        
        log_execution("Ingestor", "INFO", f"Item {item['id']} iniciado")
        
        return {
            "row_id": item['id'],
            "raw_text": item['content_text'],
            "source_type": item['source_type'],
            "status": "has_data",
            "error_count": 0
        }
    
    except Exception as e:
        log_execution("Ingestor", "ERROR", f"Erro: {str(e)}")
        return {"status": "error", "error_count": 1}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def curator_node(state: ContentState):
    """Extrai insights do conteúdo bruto"""
    agent_name = "Curator"
    log_execution(agent_name, "INFO", f"Processando item {state['row_id']}")
    
    try:
        config = get_agent_config(agent_name)
        
        messages = [
            SystemMessage(content=config['system_instruction']),
            HumanMessage(content=state['raw_text'][:10000])  # Limita tamanho
        ]
        
        response = llm.invoke(messages)
        
        log_execution(agent_name, "INFO", f"Insights extraídos com sucesso")
        return {"key_insights": response.content, "status": "curated"}
    
    except Exception as e:
        log_execution(agent_name, "ERROR", f"Erro: {str(e)}")
        return {"status": "failed", "error_count": state.get("error_count", 0) + 1}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def writer_node(state: ContentState):
    """Cria o conteúdo final baseado nos insights"""
    agent_name = "Ghostwriter"
    log_execution(agent_name, "INFO", f"Escrevendo conteúdo para item {state['row_id']}")
    
    try:
        config = get_agent_config(agent_name)
        
        prompt = f"""{config['system_instruction']}

INSIGHTS PARA TRANSFORMAR EM CONTEÚDO:
{state['key_insights']}

FONTE ORIGINAL: {state['source_type']}
"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        log_execution(agent_name, "INFO", "Conteúdo criado com sucesso")
        return {"final_draft": response.content, "status": "written"}
    
    except Exception as e:
        log_execution(agent_name, "ERROR", f"Erro: {str(e)}")
        return {"status": "failed", "error_count": state.get("error_count", 0) + 1}

def publisher_node(state: ContentState):
    """Publica o conteúdo final"""
    log_execution("Publisher", "INFO", f"Publicando item {state['row_id']}")
    
    try:
        # Salva conteúdo pronto
        supabase.table("ready_materials").insert({
            "raw_material_id": state['row_id'],
            "platform": "general",
            "final_content": state['final_draft'],
            "virality_score": 85,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        # Marca como concluído
        supabase.table("raw_materials").update({
            "status": "done",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", state['row_id']).execute()
        
        log_execution("Publisher", "SUCCESS", f"Item {state['row_id']} publicado!")
        return {"status": "finished"}
    
    except Exception as e:
        log_execution("Publisher", "ERROR", f"Erro ao publicar: {str(e)}")
        
        # Marca como failed
        supabase.table("raw_materials").update({
            "status": "failed",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", state['row_id']).execute()
        
        return {"status": "failed"}

# --- ROTEAMENTO ---
def route_ingestor(state: ContentState):
    """Decide o próximo passo após ingestão"""
    status = state.get("status")
    
    if status == "no_data":
        return END
    elif status == "error":
        return END
    else:
        return "Curator"

def route_on_failure(state: ContentState):
    """Redireciona falhas para END"""
    if state.get("status") == "failed":
        if state.get("error_count", 0) >= 3:
            log_execution("Router", "ERROR", f"Item {state['row_id']} falhou 3x. Abortando.")
            return END
    return None  # Continua normalmente

# --- MONTAGEM DO GRAFO ---
builder = StateGraph(ContentState)

# Adiciona nós
builder.add_node("Ingestor", ingest_node)
builder.add_node("Curator", curator_node)
builder.add_node("Writer", writer_node)
builder.add_node("Publisher", publisher_node)

# Define fluxo
builder.add_edge(START, "Ingestor")
builder.add_conditional_edges("Ingestor", route_ingestor)
builder.add_edge("Curator", "Writer")
builder.add_edge("Writer", "Publisher")
builder.add_edge("Publisher", END)

# Compila o grafo
content_engine = builder.compile()

# ==========================================
# PARTE 3: WORKER (LOOP DE PROCESSAMENTO)
# ==========================================
def run_agent_worker():
    """Loop infinito que processa a fila"""
    log_execution("Worker", "INFO", "🚀 Worker iniciado! Aguardando dados...")
    
    idle_count = 0
    
    while not shutdown_event.is_set():
        try:
            # Executa o pipeline
            inputs = {"status": "check", "error_count": 0}
            result = content_engine.invoke(inputs)
            
            # Verifica se processou algo
            if result.get("status") == "finished":
                log_execution("Worker", "SUCCESS", "✅ Conteúdo processado com sucesso!")
                idle_count = 0
                time.sleep(5)  # Pequena pausa entre processamentos
            
            elif result.get("status") == "no_data":
                idle_count += 1
                
                # Aumenta o tempo de espera progressivamente
                if idle_count <= 3:
                    wait_time = 10
                elif idle_count <= 10:
                    wait_time = 30
                else:
                    wait_time = 60
                
                if idle_count % 6 == 0:  # Log a cada 6 tentativas vazias
                    log_execution("Worker", "INFO", f"Fila vazia. Aguardando... ({idle_count}x)")
                
                time.sleep(wait_time)
            
            else:
                # Erro ou estado inesperado
                log_execution("Worker", "WARNING", f"Estado inesperado: {result.get('status')}")
                time.sleep(15)
        
        except Exception as e:
            log_execution("Worker", "ERROR", f"Erro crítico no worker: {str(e)}")
            time.sleep(30)  # Espera mais tempo em caso de erro
    
    log_execution("Worker", "INFO", "Worker encerrado gracefully")

# ==========================================
# PARTE 4: GRACEFUL SHUTDOWN
# ==========================================
def graceful_shutdown(sig, frame):
    """Encerra o worker de forma segura"""
    print("\n🛑 Sinal de shutdown recebido. Encerrando gracefully...")
    log_execution("System", "INFO", "Iniciando shutdown graceful")
    
    shutdown_event.set()
    
    if worker_thread and worker_thread.is_alive():
        print("⏳ Aguardando worker finalizar (max 30s)...")
        worker_thread.join(timeout=30)
        
        if worker_thread.is_alive():
            print("⚠️ Worker não finalizou a tempo. Forçando saída...")
        else:
            print("✅ Worker encerrado com sucesso")
    
    log_execution("System", "INFO", "Shutdown concluído")
    os._exit(0)

# ==========================================
# PARTE 5: INICIALIZAÇÃO
# ==========================================
if __name__ == "__main__":
    # Registra handlers de shutdown
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    
    print("=" * 60)
    print("🤖 DEEPSEEK CONTENT ENGINE v1.0")
    print("=" * 60)
    
    # Valida configurações
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("❌ DEEPSEEK_API_KEY não configurada!")
    
    log_execution("System", "INFO", "Sistema iniciando...")
    
    # Inicia worker em thread separada
    worker_thread = threading.Thread(target=run_agent_worker, daemon=True)
    worker_thread.start()
    
    print("✅ Worker iniciado em background")
    print("✅ API Flask iniciando...")
    print("=" * 60)
    
    # Inicia servidor Flask (bloqueia o processo principal)
    run_flask()