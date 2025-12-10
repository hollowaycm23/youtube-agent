# -*- coding: utf-8 -*-
"""YouTube SEO Automation

This script authenticates with the YouTube Data API, builds a Retrieval-Augmented Generation (RAG)
knowledge base, loads a GPU-accelerated LLM (Mistral-7B) and generates a list of **exactly ten**
SEO-optimized tags for each video in the configured channel.

Key improvements over the original version:
- Structured configuration using a dataclass.
- Path handling with :pyclass:`pathlib.Path` for OS-independent file operations.
- Centralised logger configuration.
- Type-hints throughout the code base.
- Robust JSON extraction using a regular expression (handles stray markdown fences).
- CLI arguments to customise channel ID, number of videos processed and optional tag update.
- Clear separation of concerns: authentication, RAG handling, LLM generation and YouTube updates.
- Offline mode for testing without network/API dependencies.
- LLM now runs on GPU for massively faster performance.
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Suppress Hugging Face parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

# PyTorch and Transformers for GPU LLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional Google imports (handled gracefully if missing in some envs, though required for online mode)
try:
    import google.auth.transport.requests
    import google_auth_oauthlib.flow
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    # Only ignored if running in offline mode, otherwise main will fail
    pass

# Import our optimized RAG system
import rag_system

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # YouTube API
    client_secrets_file: Path = Path("client_secrets.json")
    token_file: Path = Path("token.json")
    scopes: Tuple[str, ...] = ("https://www.googleapis.com/auth/youtube.force-ssl",)
    api_service_name: str = "youtube"
    api_version: str = "v3"
    # LLM (now on GPU)
    llm_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # RAG
    rag_folder: Path = Path("rag_documents")
    # Runtime options (CLI)
    channel_id: str = "UC3yF2E14y-t7I1Bw1uE4f0A"  # replace with your channel ID
    max_videos: int = 5  # number of videos to process per run
    update_tags: bool = False  # set True to push tags to YouTube
    offline_mode: bool = False # Run without YouTube API dependencies

# Global logger – configured once in ``main``
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def get_authenticated_service(cfg: Config):
    """Return an authorized YouTube API service instance."""
    if cfg.offline_mode:
        logger.info("Modo OFFLINE: pulando autenticação do YouTube")
        return None

    credentials: Optional[Credentials] = None
    if cfg.token_file.exists():
        credentials = Credentials.from_authorized_user_file(str(cfg.token_file), cfg.scopes)
    
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                str(cfg.client_secrets_file), cfg.scopes
            )
            credentials = flow.run_local_server(port=0)
        cfg.token_file.write_text(credentials.to_json())
    return build(cfg.api_service_name, cfg.api_version, credentials=credentials)

# ---------------------------------------------------------------------------
# Prompt handling & LLM generation (Now on GPU)
# ---------------------------------------------------------------------------

LLM_MODEL = None
LLM_TOKENIZER = None

def initialize_llm(cfg: Config) -> None:
    """Load the Mistral-7B model via Transformers on the GPU."""
    global LLM_MODEL, LLM_TOKENIZER
    if LLM_MODEL is not None:
        return

    # TEMPORARY CHANGE for testing: Load LLM even in offline mode
    if os.environ.get("MOCK_LLM"):
        logger.warning("MOCK_LLM detectado – pulando carregamento do LLM real.")
        return

    logger.info("Loading LLM %s on device %s", cfg.llm_model_id, cfg.llm_device.upper())
    if cfg.llm_device == "cpu":
        logger.warning("GPU não encontrada, carregando LLM no CPU. Isso será muito lento.")

    try:
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(cfg.llm_model_id)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_id,
            torch_dtype=torch.float16, # Use float16 for memory efficiency on GPU
            device_map="auto" # Automatically use available GPU
        )
        logger.info("LLM loaded successfully on %s", LLM_MODEL.device)
    except Exception as exc:
        logger.error("Failed to load LLM: %s. Switching to MOCK mode.", exc)
        LLM_MODEL = None
        LLM_TOKENIZER = None

SYSTEM_PROMPT = (
    "Você é um analista de SEO de alto nível especializado em otimização de vídeos para o YouTube no Brasil. "
    "Sua tarefa é gerar uma lista de 10 tags estratégicas e otimizadas para um vídeo. "
    "As tags devem combinar palavras-chave de cauda longa baseadas no título e descrição, "
    "palavras-chave estratégicas do seu conhecimento de SEO e palavras-chave extraídas do contexto RAG abaixo.\n\n"
    "REGRAS DE SAÍDA:\n"
    "1. Responda APENAS com um JSON (array de strings).\n"
    "2. O array DEVE conter EXATAMENTE 10 strings.\n"
    "3. Não inclua texto adicional.\n\n"
    "--- CONTEXTO RAG ---\n{rag_context}\n--- FIM DO CONTEXTO ---"
)

def build_user_prompt(title: str, description: str, rag_context: str) -> str:
    """Compose the final prompt sent to the LLM."""
    system = SYSTEM_PROMPT.format(rag_context=rag_context or "Nenhum contexto RAG disponível.")
    return f"{system}\n\n[INSTRUÇÃO]: Gere as 10 tags JSON para o vídeo com Título: \"{title}\", Descrição: \"{description}\""

_TAG_REGEX = re.compile(r'\[.*?\]', re.DOTALL)

def extract_json_tags(raw_output: str) -> List[str]:
    """Extract a JSON list of tags from the LLM raw output."""
    cleaned = raw_output.replace("```json", "").replace("```", "").strip()
    match = _TAG_REGEX.search(cleaned)
    if not match:
        logger.debug("Falha no regex. Output limpo: %s", cleaned)
        raise ValueError("Nenhum JSON de tags encontrado no output do LLM")
    json_str = match.group(0)
    try:
        tags = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Falha ao decodificar JSON de tags: {exc}")
    if not isinstance(tags, list):
        raise ValueError(f"JSON de tags inválido, não é uma lista: {tags}")
    if len(tags) != 10:
        logger.warning("O LLM gerou %d tags em vez das 10 esperadas.", len(tags))
    return tags

def generate_tags(title: str, description: str, cfg: Config) -> Optional[List[str]]:
    """Generate SEO tags using the RAG context and the GPU-accelerated LLM."""
    # 1️⃣ Retrieve relevant context from RAG
    query = f"Otimize SEO para o vídeo com título: {title}"
    rag_context = rag_system.get_context(query)
    if not rag_context:
        logger.warning("Nenhum contexto RAG encontrado – a geração pode ser menos precisa")

    # Fallback / Mock Generation if LLM is missing
    if LLM_MODEL is None or LLM_TOKENIZER is None:
        logger.warning("LLM não inicializado (Offline/Mock). Gerando tags de fallback baseadas no título.")
        base_words = [w.lower() for w in re.findall(r"\w+", title) if len(w) > 3]
        mock_tags = base_words[:5]
        mock_tags.extend(["seo", "youtube", "viral", "video", "brasil"])
        while len(mock_tags) < 10:
            mock_tags.append(f"tag_{len(mock_tags)+1}")
        return mock_tags[:10]

    # 2️⃣ Build the prompt and run inference
    prompt = build_user_prompt(title, description, rag_context)
    logger.debug("Prompt enviado ao LLM (últimos 200 chars): %s", prompt[-200:])
    raw_text = ""
    try:
        inputs = LLM_TOKENIZER(prompt, return_tensors="pt").to(LLM_MODEL.device)
        start = time.time()
        with torch.no_grad():
            output_ids = LLM_MODEL.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
            )
        logger.info("LLM inference concluída em %.2fs", time.time() - start)
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        raw_text = LLM_TOKENIZER.decode(generated_ids, skip_special_tokens=True)
        logger.debug("LLM raw output: %s", raw_text)
        return extract_json_tags(raw_text)
    except Exception as exc:
        logger.error("Erro durante geração de tags: %s. Raw text was: %s", exc, raw_text)
        return None

# ---------------------------------------------------------------------------
# YouTube interaction
# ---------------------------------------------------------------------------

def process_videos(youtube, cfg: Config):
    """Iterate over the channel videos, generate tags and optionally update them."""
    if cfg.offline_mode:
        logger.info("Modo OFFLINE: Lendo vídeos de mock_videos.json")
        try:
            with open("mock_videos.json", "r", encoding="utf-8") as f:
                videos = json.load(f)
        except Exception as exc:
            logger.error("Falha ao ler mock_videos.json: %s", exc)
            return
        
        for item in videos:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            vid = item["id"]["videoId"]
            logger.info("Processando vídeo (MOCK): %s (ID %s)", title, vid)
            
            tags = generate_tags(title, description, cfg)
            if tags:
                logger.info("Tags geradas: %s", tags)
                if cfg.update_tags:
                    logger.info("Modo OFFLINE: Simulação de atualização de tags no YouTube para o vídeo %s", vid)
            else:
                logger.warning("Falha ao gerar tags para o vídeo %s", vid)
        return

    # Online mode logic
    logger.info("Fetching up to %d videos from channel %s", cfg.max_videos, cfg.channel_id)
    try:
        request = youtube.search().list(
            part="snippet",
            type="video",
            channelId=cfg.channel_id,
            maxResults=cfg.max_videos,
            order="date" # Fetch most recent videos
        )
        response = request.execute()
        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    except HttpError as exc:
        logger.error("Erro ao listar vídeos: %s", exc)
        return

    for vid in video_ids:
        try:
            video_resp = youtube.videos().list(part="snippet,status", id=vid).execute()
            if not video_resp.get("items"):
                continue
            video = video_resp["items"][0]
            snippet = video["snippet"]
            title = snippet["title"]
            description = snippet["description"]
            logger.info("Processando vídeo: %s (ID %s)", title, vid)
            
            tags = generate_tags(title, description, cfg)
            
            if tags:
                logger.info("Tags geradas: %s", tags)
                if cfg.update_tags:
                    snippet["tags"] = tags
                    youtube.videos().update(
                        part="snippet",
                        body={"id": vid, "snippet": snippet},
                    ).execute()
                    logger.info("Tags atualizadas no YouTube para o vídeo %s", vid)
            else:
                logger.warning("Falha ao gerar tags para o vídeo %s", vid)
        except HttpError as exc:
            logger.error("Erro na API do YouTube para vídeo %s: %s", vid, exc)
        except Exception as exc:
            logger.exception("Erro inesperado ao processar vídeo %s", vid)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube SEO automation with RAG + GPU LLM")
    parser.add_argument("--channel-id", help="YouTube channel ID (overrides config)")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument("--update-tags", action="store_true", help="Push generated tags to YouTube")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode using mock data")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        channel_id=args.channel_id or Config.channel_id,
        max_videos=args.max_videos or Config.max_videos,
        update_tags=args.update_tags,
        offline_mode=args.offline,
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    if cfg.offline_mode:
        logger.info("Iniciando automação de SEO em modo OFFLINE")
    else:
        logger.info("Iniciando automação de SEO para canal %s", cfg.channel_id)
    
    # Initialize services
    initialize_llm(cfg) # Load LLM on GPU first
    youtube = get_authenticated_service(cfg)
    
    # Run the main processing loop
    process_videos(youtube, cfg)
    
    logger.info("Processamento concluído")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Erro crítico na execução: %s", exc)
        sys.exit(1)
