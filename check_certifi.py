import ssl
import warnings
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.utils import CryptographyDeprecationWarning

def find_bad_certs():
    print("正在扫描Windows证书存储中的问题证书...\n")
    for store_name in ["ROOT", "CA"]:
        print(f"=== 检查存储区: {store_name} ===")
        try:
            for cert, encoding, trust in ssl.enum_certificates(store_name):
                if encoding == "x509_asn":
                    # 使用 warnings.catch_warnings 捕获 CryptographyDeprecationWarning
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            cert_obj = x509.load_der_x509_certificate(cert, default_backend())
                            # 检查是否有警告且警告消息包含 "serial number which wasn't positive"
                            if w and any("serial number which wasn't positive" in str(warn.message) for warn in w):
                                print(f"\n[!] 发现序列号非法的证书:")
                                print(f"   颁发者: {cert_obj.issuer}")
                                print(f"   主题: {cert_obj.subject}")
                                print(f"   序列号: {cert_obj.serial_number} (注意: 此序列号为负数或零)")
                        except Exception as e:
                            print(f"\n[!] 发现损坏的证书，错误: {e}")
                            hex_preview = cert[:20].hex()
                            print(f"   证书原始数据预览(前20字节): {hex_preview}")
        except PermissionError:
            print(f"  无法访问存储区 {store_name}，请以管理员身份运行此脚本。")

if __name__ == "__main__":
    find_bad_certs()