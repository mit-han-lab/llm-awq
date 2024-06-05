import os
import re


## ------ Install Pytorch ------- ##
# Detect the current cuda version
version = os.popen('nvcc --version').read()

pattern = r'release (\d+\.\d+)'
match = re.search(pattern, version)
if match:
    cuda_version = match.group(1)
else:
    cuda_version = None
    raise ValueError('Could not detect cuda version')

cuda_major_version = cuda_version.split('.')[0]

if cuda_major_version == '11':
    os.system('pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118')
    os.system('wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')
    os.system('pip install flash_attn-2.4.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')
elif cuda_major_version == '12':
    os.system('pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121')
    os.system('wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')
    os.system('pip install flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl')
else:
    raise ValueError(f'Unsupported cuda version {cuda_version}')


## ------ Install AWQ ------- ##
os.system('pip install -e .')