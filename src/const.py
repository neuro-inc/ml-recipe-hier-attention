from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = Path(ROOT_PATH / 'data')
RESULT_DIR = ROOT_PATH / 'results'

VECTORS_CACHE = RESULT_DIR / '.vector_cache'
IMBD_ROOT = DATA_PATH / 'aclImdb'
LOG_DIR = RESULT_DIR / 'logs'
