"""Configure pytest for the test suite."""
import warnings

from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest."""
    # Filter out the setuptools deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="pkg_resources is deprecated as an API.*",
        module="pkg_resources.*",
    ) 