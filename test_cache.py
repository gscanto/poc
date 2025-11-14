import requests
import time

API_URL = "http://localhost:8000"

# Dados de teste
test_data = {
    "title": "Teste de Cache",
    "context": "Este é um contexto de teste para verificar o cache.",
    "sections": ["Introdução", "Metodologia", "Resultados"]
}

print("Enviando primeira requisição...")
start = time.time()
response1 = requests.post(f"{API_URL}/generate-report", json=test_data)
end = time.time()
print(f"Primeira resposta: {response1.status_code}, Tempo: {end - start:.2f}s")

if response1.status_code == 200:
    data1 = response1.json()
    print(f"ID: {data1['report_id']}, Tempo geração: {data1['generation_time']}")

print("\nEnviando segunda requisição (mesmo contexto)...")
start = time.time()
response2 = requests.post(f"{API_URL}/generate-report", json=test_data)
end = time.time()
print(f"Segunda resposta: {response2.status_code}, Tempo: {end - start:.2f}s")

if response2.status_code == 200:
    data2 = response2.json()
    print(f"ID: {data2['report_id']}, Tempo geração: {data2['generation_time']}")

    # Verificar se conteúdo é igual
    if data1['content'] == data2['content']:
        print("Conteúdo idêntico - cache funcionando!")
    else:
        print("Conteúdo diferente - erro no cache")
