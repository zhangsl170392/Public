import ssl
import certifi

# 保存原始函数备用（其实用不上）
original_create = ssl.create_default_context

def patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH,
                                   cafile=None, capath=None, cadata=None):
    """
    自定义创建 SSL 上下文，绕开 Windows 系统证书存储，
    直接使用 certifi 提供的证书包。
    """
    # 选择协议：客户端或服务端
    if purpose == ssl.Purpose.SERVER_AUTH:
        protocol = ssl.PROTOCOL_TLS_CLIENT
    else:
        protocol = ssl.PROTOCOL_TLS_SERVER

    context = ssl.SSLContext(protocol)

    # 如果传入了 cafile/capath/cadata，优先使用它们
    if cafile:
        context.load_verify_locations(cafile)
    elif capath:
        context.load_verify_locations(capath=capath)
    elif cadata:
        context.load_verify_locations(cadata=cadata)
    else:
        # 否则使用 certifi 的证书文件（这是关键）
        context.load_verify_locations(cafile=certifi.where())

    # 如果不希望验证主机名（开发环境），可取消下面注释（但建议保留）
    # context.check_hostname = False
    return context

# 替换全局函数，影响所有后续导入（包括 tornado）
ssl.create_default_context = patched_create_default_context

# 现在再导入 streamlit 的 CLI
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8888"]
    main()