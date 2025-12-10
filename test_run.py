import sys
import os
import logging
# Ensure the project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# We need online access to download models (at least once)
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]

import youtube_seo_automation as ysa
import rag_system

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("test_run")
    
    logger.info("Starting test run...")
    
    # Use a Config with no video processing to avoid API calls
    cfg = ysa.Config(max_videos=0, update_tags=False, offline_mode=True)
    
    # Force rebuild of DB to ensure new embedding model is used
    rag_index = "rag_faiss.index"
    rag_meta = "rag_metadata.json"
    
    if os.path.exists(rag_index) or os.path.exists(rag_meta):
        logger.info("Removing old RAG files to force re-embedding")
        if os.path.exists(rag_index): os.remove(rag_index)
        if os.path.exists(rag_meta): os.remove(rag_meta)
    
    # Note: we don't need to manually load the RAG DB. 
    # ysa.generate_tags calls rag_system.get_context, which handles loading/building.
    
    # Initialize the LLM (OpenVINO)
    # The user might want to skip this if strictly testing RAG, but let's try to run it.
    # If connection fails or model missing, it will fallback to mock.
    logger.info("Initializing LLM...")
    ysa.initialize_llm(cfg)
    
    # Generate tags for a dummy query
    logger.info("Generating tags for test video...")
    tags = ysa.generate_tags("Teste de vídeo sobre Python", "Um vídeo ensinando automação com Python e RAG.", cfg)
    logger.info(f"Generated tags: {tags}")

if __name__ == "__main__":
    main()
