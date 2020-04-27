from .samesizekmeans import SameSizeKmeans

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "'pandas<1.0'"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

