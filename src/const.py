from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data'

VECTORS_CACHE = DATA_PATH / '.vector_cache'
IMBD_ROOT = DATA_PATH / 'aclImdb'
LOG_DIR = DATA_PATH.parent / 'results' / 'logs'
LOG_DIR_CATALYST = DATA_PATH.parent / 'results' / 'catalyst_logs'

WANDB_TOKEN_FILE = Path(__file__).parent / 'wandb_token.txt'
