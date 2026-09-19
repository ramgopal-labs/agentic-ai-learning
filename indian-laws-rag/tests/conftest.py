import sys
from pathlib import Path

import pytest

# Make `import app...` work when pytest is run from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def override_settings():
    """
    Temporarily change frozen `settings` fields, restoring them afterwards.

    Settings is a frozen dataclass, so monkeypatch cannot set attributes on it.
    Every module binds the same single instance, so writing through
    object.__setattr__ is visible everywhere it was imported.
    """
    from app.core.config import settings

    originals: dict[str, object] = {}

    def _override(**values):
        for name, value in values.items():
            if name not in originals:
                originals[name] = getattr(settings, name)
            object.__setattr__(settings, name, value)

    yield _override

    for name, value in originals.items():
        object.__setattr__(settings, name, value)
