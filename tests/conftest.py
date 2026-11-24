import os
import tempfile
import pytest
from core.database import init_db, seed_demo_data
from core.config import get_settings


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database pre-seeded with demo data for isolated testing."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    
    init_db(path)
    seed_demo_data(path, force=True)
    
    yield path
    
    # Cleanup after test
    try:
        if os.path.exists(path):
            os.remove(path)
        for ext in ("-wal", "-shm"):
            if os.path.exists(path + ext):
                os.remove(path + ext)
    except OSError:
        pass
