"""Print the configured nginx URL using the same include-aware reader as /doctor."""
import os

from .preflight import _nginx_site


def nginx_message(port: str) -> str:
    site = _nginx_site()
    if not site or not site["proxy_port"] or not site["host"] or not site["listen_port"]:
        return "  區網(nginx)   無法判讀代理設定；請用 ./run.sh --doctor 檢查。"
    if site["proxy_port"] != port:
        return (f"  區網(nginx)   ✗ 代理指到 :{site['proxy_port']},但面板綁的是 :{port}。\n"
                "                 修:sudo scripts/deploy-nginx-lan.sh")
    return (f"  區網(nginx)   https://{site['host']}{site['url_port']}/"
            "   ← 給同事的就是這個(要帳密)")


if __name__ == "__main__":
    print(nginx_message(os.environ.get("RECON_STUDIO_PORT", "8077")))
