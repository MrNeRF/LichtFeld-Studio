"""Package management via uv"""



class PackageInfo:
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def path(self) -> str: ...

    def __repr__(self) -> str: ...

def init() -> str:
    """Initialize venv at ~/.lichtfeld/venv"""

def install(package: str) -> str:
    """Install package from PyPI"""

def uninstall(package: str) -> str:
    """Uninstall package"""

def list() -> list[PackageInfo]:
    """List installed packages"""

def is_installed(package: str) -> bool:
    """Check if package is installed"""

def is_uv_available() -> bool:
    """Check if uv is available"""

def uv_path() -> str:
    """Get path to uv binary (empty string if not found)"""

def embedded_python_path() -> str:
    """Get path to embedded Python executable (empty string if not available)"""

def install_torch(cuda: str = 'auto', version: str = '') -> str:
    """Install PyTorch with CUDA detection"""

def site_packages_dir() -> str:
    """Get site-packages path"""

def install_async(package: str) -> bool:
    """Install package asynchronously (non-blocking)"""

def install_torch_async(cuda: str = 'auto', version: str = '') -> bool:
    """Install PyTorch asynchronously (non-blocking)"""

def is_busy() -> bool:
    """Check if async operation is running"""

def typings_dir() -> str:
    """Get path to type stubs directory (empty if not found)"""
