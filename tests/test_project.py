from phishguard import __version__
from phishguard.config import ARTIFACTS_DIR, PROJECT_ROOT, RAW_DATA_DIR


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_project_paths_are_inside_repository() -> None:
    assert RAW_DATA_DIR == PROJECT_ROOT / "data" / "raw"
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"
