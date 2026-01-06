import os
from dotenv import load_dotenv
from supabase import create_client
import requests
import time
import urllib.parse

# Carrega ambiente
load_dotenv()

# Configuração
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "content-images"

print(f"--- DIAGNÓSTICO DE IMAGEM ---")
print(f"1. Conectando ao Supabase...")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase conectado.")
except Exception as e:
    print(f"❌ Erro ao conectar Supabase: {e}")
    exit()

def test_pipeline():
    # 1. Testar Geração (Pollinations)
    print("\n2. Testando Pollinations API...")
    prompt = "futuristic cyberpunk city neon lights, high detail, night scene  artstation trending" 
    encoded_prompt = urllib.parse.quote(prompt)
    seed = 42
    
    # URL do Pollinations
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=768&seed={seed}&model=flux&nologo=true"
    print(f"   URL Gerada: {image_url}")
    
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            print(f"✅ Download da imagem com sucesso ({len(response.content)} bytes).")
        else:
            print(f"❌ Falha no download. Status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Erro de conexão com Pollinations: {e}")
        return

    # 2. Testar Upload (Supabase)
    print("\n3. Testando Upload para Supabase Storage...")
    file_name = f"test_image_{int(time.time())}.png"
    
    try:
        # Tenta upload
        res = supabase.storage.from_(BUCKET_NAME).upload(
            path=file_name,
            file=response.content,
            file_options={"content-type": "image/png"}
        )
        print(f"✅ Upload realizado! Caminho: {res}")
        
        # Tenta pegar URL pública
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        print(f"\n🎉 SUCESSO TOTAL! Sua URL final:")
        print(public_url)
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO NO UPLOAD: {e}")
        print("DICA: Verifique se o Bucket 'content-images' existe e se as Policies permitem INSERT/UPLOAD.")

if __name__ == "__main__":
    test_pipeline()