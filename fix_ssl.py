import os
import certifi
import ssl

# 1. 设置环境变量（辅助作用）
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

# 2. 保存原始函数
_original_create_default_context = ssl.create_default_context

# 3. 定义新的函数，强制使用 certifi 的证书文件，从而绕过 Windows 存储
def _patched_create_default_context(*args, **kwargs):
    # 如果调用者没有指定任何证书源，则强制指定 certifi 的证书文件
    if 'cafile' not in kwargs and 'capath' not in kwargs and 'cadata' not in kwargs:
        kwargs['cafile'] = certifi.where()
    return _original_create_default_context(*args, **kwargs)

# 4. 替换全局函数
ssl.create_default_context = _patched_create_default_context