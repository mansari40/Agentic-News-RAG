import logging
import sys

from rag_baseline.ingestion.article_ingestor import IngestionService
from rag_baseline.structured_logging import configure_logging

if __name__ == "__main__":
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 60)
        logger.info("STARTING INGESTION PIPELINE")
        logger.info("=" * 60)

        pipeline = IngestionService()
        pipeline.ingest(days_back=85)

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("INGESTION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
