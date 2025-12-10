# YouTube SEO Automation Agent Instructions

## Project Overview

This is a **YouTube channel SEO optimization system** combining three architectural layers:

1. **RAG (Retrieval-Augmented Generation)** (`rag_system.py`) - Vector database that embeds documents (PDFs, docs, markdown, video transcriptions) for context-aware prompt enrichment
2. **LLM Generation** (`youtube_seo_automation.py`) - OpenVINO-optimized Mistral-7B model generating exactly 10 SEO-optimized tags
3. **YouTube API Integration** - Fetches videos, processes them, optionally pushes generated tags back to YouTube

## Architecture & Data Flow

### Core Pipeline
```
Channel Videos → Title+Description → RAG Retrieval (semantic search) 
→ LLM Prompt (system + RAG context) → Tag Extraction (JSON regex) 
→ YouTube Update (if enabled)
```

### Key Components

**`youtube_seo_automation.py`** (Main Entry Point)
- `Config` dataclass: Frozen configuration with YouTube credentials, LLM settings, RAG folder path
- `get_authenticated_service()`: OAuth2 flow with token persistence (`token.json`)
- `initialize_llm()`: Lazy-loads Mistral via `optimum.intel.openvino`, returns (model, tokenizer) tuple
- `generate_tags()`: Orchestrates RAG retrieval → prompt building → LLM inference → JSON extraction
- `process_videos()`: Dual-mode loop (online via YouTube API or offline using `mock_videos.json`)
- Fallback tags generated if LLM is unavailable (title-based words + generic tags)

**`rag_system.py`** (Knowledge Base)
- `initialize_models()`: Loads embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 384D→1024D) and Whisper-tiny
- `create_vector_db()`: Scans `rag_documents/` folder, extracts text (supports .txt, .md, .pdf, .docx, .xlsx), transcribes media (.mp4, .mp3), caches embeddings in `rag_vector_db.json`
- `calculate_embedding()`: Normalizes embeddings; falls back to SHA256 hash-based deterministic vectors if model unavailable
- `retrieve_context()`: Cosine similarity retrieval with configurable threshold (0.7) and 4000-token context limit
- FastAPI optional server (`/query` endpoint) for external integration
# YouTube SEO Automation — Instruções para Agentes (pt-BR)

Objetivo: ajudar um agente de codificação a ficar produtivo rapidamente neste repositório, documentando arquitetura, fluxos de trabalho e padrões específicos do projeto.

**Visão geral**
- Pipeline: `vídeos do canal` → extrair título+descrição → `rag_system` recupera contexto → `youtube_seo_automation.generate_tags()` monta o prompt → LLM → extração JSON de tags → opcional `--update-tags` para enviar ao YouTube.
- Componentes principais: `youtube_seo_automation.py` (orquestração + YouTube API), `rag_system.py` (DB vetorial RAG + transcrições), `rag_documents/` (documentos fonte), `mock_videos.json` (dados de teste offline), `rag_vector_db.json` (embeddings em cache).

**Comandos rápidos (execução e desenvolvimento)**
```bash
# Instalar dependências e preparar ambiente
bash setup_and_run.sh

# Teste offline (sem YouTube API ou downloads do HF)
HF_HUB_OFFLINE=1 MOCK_LLM=1 python3 youtube_seo_automation.py --offline --log-level DEBUG

# Recriar banco vetorial após adicionar documentos
python3 rag_system.py --add path/to/file.pdf

# Executar testes leves
python3 test_run.py
```

**Convenções e padrões do repositório**
- Carregamento preguiçoso de modelos: embedding/whisper/LLM são inicializados sob demanda (veja `initialize_models()` e `initialize_llm()`). O código verifica variáveis `None` e usa alternativas quando necessário.
- Fallbacks determinísticos: `calculate_embedding()` pode gerar vetores determinísticos (baseados em SHA256) quando o modelo de embedding não está disponível — isso mantém comportamento reprodutível offline.
- Formato de saída exato: `generate_tags()` / `extract_json_tags()` espera exatamente 10 tags. Se a extração JSON falhar, o código usa uma lista de fallback derivada do título — não altere esse contrato sem atualizar a lógica de fallback.
- Configuração: o dataclass `Config` é `frozen` e centraliza opções de execução; argumentos CLI sobrescrevem os padrões.

**Arquivos e funções importantes para alterações**
- `youtube_seo_automation.py`: `get_authenticated_service()`, `initialize_llm()`, `generate_tags()`, `process_videos()` — mudanças de prompt, parâmetros do LLM ou comportamento de atualização ficam aqui.
- `rag_system.py`: `create_vector_db()`, `calculate_embedding()`, `retrieve_context()`, `extract_text_from_file()` — aqui se alteram formatos suportados, modelo de embedding e parâmetros de busca semântica.
- `setup_and_run.sh` e `requirements.txt`: setup de ambiente e dependências. CI/repos contribuintes devem usar estes arquivos.

**Flags de ambiente e regras de segurança**
- `--offline` evita chamadas ao YouTube e usa `mock_videos.json`.
- `--update-tags` é necessário para enviar tags ao YouTube; exige `client_secrets.json` e gera/usa `token.json`. Não ative `--update-tags` sem autorização explícita do dono do canal.
- Variáveis úteis: `HF_HUB_OFFLINE=1` (evita downloads do Hugging Face), `MOCK_LLM=1` (usa comportamento de LLM mockado).

**Ao editar prompts, modelos ou o RAG**
- Afinar regras de SEO: editar `SYSTEM_PROMPT` em `youtube_seo_automation.py`.
- Trocar modelo de embedding: atualizar `EMBEDDING_MODEL_ID` em `rag_system.py` e remover `rag_vector_db.json` para evitar vetores obsoletos.
- Adicionar tipos de arquivo: estender `supported_exts` e `extract_text_from_file()` em `rag_system.py`.

**Exemplos (copiar/colar)**
- Geração offline (sem autenticação):
```bash
HF_HUB_OFFLINE=1 MOCK_LLM=1 python3 youtube_seo_automation.py --offline --log-level DEBUG
```
- Recriar vetores após adicionar documentos:
```bash
python3 rag_system.py --add rag_documents/arquivo.pdf
```

**Checklist de segurança e testes para agentes**
- Não chame APIs ao vivo a menos que o usuário tenha solicitado `--update-tags` e fornecido `client_secrets.json`.
- Preserve o comportamento offline e os fallbacks (`MOCK_LLM`, `HF_HUB_OFFLINE`).
- Se alterar o número de tags de saída, atualize todas as chamadas e o fallback correspondente.

Posso também:
- Adicionar um `README.md` curto com esses comandos
- Criar testes unitários para `extract_json_tags()` e para o contrato de `retrieve_context()` no RAG

Por favor revise esta versão em português e diga quais seções você quer expandir ou ajustar.
    Purpose: help an AI coding agent be productive in this repo (YouTube SEO automation).
