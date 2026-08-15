import logging

logging.basicConfig(
  level = logging.INFO,
  format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

logger.debug("This is debug information")
logger.info("Model evaluation started")
logger.warning("Model accuracy is below expected threshold")
logger.error("Model evaluation failed")