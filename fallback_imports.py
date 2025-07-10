# Fallback imports for testing environment
import sys
from unittest.mock import Mock

# Mock requests if not available
try:
    import requests
except ImportError:
    requests = Mock()
    requests.post = Mock(return_value=Mock(status_code=200, json=lambda: {}))
    requests.get = Mock(return_value=Mock(status_code=200, json=lambda: {}))
    sys.modules['requests'] = requests

# Mock psutil if not available  
try:
    import psutil
except ImportError:
    psutil = Mock()
    psutil.virtual_memory = Mock(return_value=Mock(total=8*1024**3))  # 8GB
    psutil.cpu_count = Mock(return_value=8)
    psutil.disk_usage = Mock(return_value=Mock(free=100*1024**3))  # 100GB
    sys.modules['psutil'] = psutil

# Mock aiohttp if not available
try:
    import aiohttp
except ImportError:
    aiohttp = Mock()
    sys.modules['aiohttp'] = aiohttp
