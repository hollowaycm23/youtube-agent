# YouTube SEO Automation

Resumos rápidos e comandos para desenvolvimento e execução deste repositório.

Comandos importantes:

```bash
# Instalar dependências (Linux):
bash setup_and_run.sh

# Executar em modo offline (usa `mock_videos.json` e não baixa modelos):
HF_HUB_OFFLINE=1 MOCK_LLM=1 python3 youtube_seo_automation.py --offline --log-level DEBUG

# Recriar vetores após adicionar documentos em `rag_documents/`:
python3 rag_system.py --add rag_documents/arquivo.pdf

# Executar o teste leve incluído (não requer pytest):
python3 test_run.py

# Executar os testes unitários (unittest):
python3 -m unittest discover -v
```

Pontos-chave para agentes e contribuintes:

- O pipeline principal está em `youtube_seo_automation.py`.
- O RAG está em `rag_system.py` e armazena embeddings em `rag_vector_db.json`.
- Use `--offline` para evitar chamadas ao YouTube e para desenvolvimento sem credenciais.
- Nunca rode com `--update-tags` sem ter `client_secret.json` e autorização explícita do dono do canal.

Se quiser que eu inclua instruções de CI ou `requirements.txt` atualizados para `unittest/pytest`, diga qual abordagem prefere.
