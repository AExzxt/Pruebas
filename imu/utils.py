import logging
import signal
import sys
from typing import Callable, Optional

def clamp(val: float, min_val: float, max_val: float) -> float:
    """Limita un valor entre min y max."""
    return max(min(val, max_val), min_val)

def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level
    )

def handle_signals(cleanup: Callable[[], None], logger: Optional[logging.Logger] = None):
    """Registra manejadores para SIGINT/SIGTERM para parada limpia."""
    def _handler(signum, frame):
        if logger:
            logger.info(f"Señal recibida: {signum}. Cerrando...")
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
