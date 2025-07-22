import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

# --- 配置部分 ---
# 根据你的 Clash 配置更新
CLASH_API_URL = "http://127.0.0.1:10090"  # <--- **已更新为 9090**
CLASH_MIXED_PORT = 17790              # <--- **已更新为 7890**

# 你的 Clash 配置中没有 secret，所以这里保持为空字符串
CLASH_API_SECRET = ''

# 要测试的目标网站
TARGET_URL = "https://api.wandb.ai"

# 要测试的代理组名称 (根据你的 Clash 配置，这是 "♻️ 自动选择")
TARGET_PROXY_GROUP = "♻️ 自动选择"
ENCODED_TARGET_PROXY_GROUP = quote(TARGET_PROXY_GROUP) # 用于 URL 路径编码

# 测试每个节点时的超时时间（秒）
TEST_TIMEOUT = 10
# 切换代理后等待的时间（秒），给 Clash 一点时间切换和生效
SWITCH_DELAY = 1.5 # 稍微增加一点等待时间，确保切换完成

# --- 核心函数 ---
def get_clash_proxies_info():
    """
    从 Clash API 获取所有代理和代理组的详细信息。
    """
    headers = {}
    if CLASH_API_SECRET:
        headers['Authorization'] = f'Bearer {CLASH_API_SECRET}'

    try:
        response = requests.get(f"{CLASH_API_URL}/proxies", headers=headers, timeout=5)
        response.raise_for_status() # 如果状态码不是2xx，将抛出HTTPError
        return response.json().get('proxies', {}) # 返回 'proxies' 键下的对象
    except requests.exceptions.RequestException as e:
        print(f"错误：无法从 Clash API 获取代理信息。请确保 Clash 控制器运行在 {CLASH_API_URL}。")
        print(f"详细错误: {e}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"API 响应内容 (HTTP {e.response.status_code}): {e.response.text}")
        return None

def set_clash_proxy_for_group(group_name_encoded, proxy_name_to_select):
    """
    通过 Clash API 设置指定代理组的当前代理。
    """
    url = f"{CLASH_API_URL}/proxies/{group_name_encoded}"
    headers = {'Content-Type': 'application/json'}
    if CLASH_API_SECRET:
        headers['Authorization'] = f'Bearer {CLASH_API_SECRET}'

    payload = {"name": proxy_name_to_select}
    try:
        response = requests.put(url, headers=headers, data=json.dumps(payload), timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"警告：无法将组 '{group_name_encoded}' 切换到代理 '{proxy_name_to_select}'。错误: {e}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"API 响应内容 (HTTP {e.response.status_code}): {e.response.text}")
        return False

def test_connectivity_via_clash(target_url, clash_mixed_port):
    """
    通过 Clash 的混合代理端口测试目标URL的连接性。
    """
    proxies = {
        "http": f"http://127.0.0.1:{clash_mixed_port}",
        "https": f"http://127.0.0.1:{clash_mixed_port}",
    }
    try:
        # 使用 HEAD 请求验证连接，因为它只请求头部信息，速度更快
        # verify=True 是默认值，保持开启以验证SSL证书。
        response = requests.head(target_url, proxies=proxies, timeout=TEST_TIMEOUT)
        return response.status_code == 200 # 检查HTTP状态码是否为200 (OK)
    except requests.exceptions.RequestException as e:
        # print(f"    测试连接失败: {e}") # 可以在调试时取消注释
        return False

def main():
    print(f"正在从 Clash API ({CLASH_API_URL}) 获取所有代理信息...")
    all_proxies_data = get_clash_proxies_info()

    if not all_proxies_data:
        return

    # 找到目标代理组 '♻️ 自动选择' 的信息
    target_group_info = all_proxies_data.get(TARGET_PROXY_GROUP)

    if not target_group_info or 'all' not in target_group_info:
        print(f"错误：在 Clash API 中未找到代理组 '{TARGET_PROXY_GROUP}' 或其不包含 'all' 列表。")
        print("请确认你的 Clash 配置中存在此代理组，且它是一个 url-test 类型组。")
        return

    # 获取该组下所有可选择的代理名称
    selectable_proxy_names = target_group_info['all']
    if not selectable_proxy_names:
        print(f"代理组 '{TARGET_PROXY_GROUP}' 中没有可选择的代理节点。")
        return

    print(f"在 '{TARGET_PROXY_GROUP}' 组中找到 {len(selectable_proxy_names)} 个节点。开始逐一测试对 {TARGET_URL} 的连接...")

    successful_nodes = []

    for i, proxy_name in enumerate(selectable_proxy_names):
        print(f"\n--- 测试节点 {i+1}/{len(selectable_proxy_names)}: {proxy_name} ---")

        # 1. 切换 Clash 代理组到当前节点
        print(f"正在切换 '{TARGET_PROXY_GROUP}' 组到 '{proxy_name}'...")
        if not set_clash_proxy_for_group(ENCODED_TARGET_PROXY_GROUP, proxy_name):
            print(f"[❌ 失败] 无法切换到节点 '{proxy_name}'。跳过测试。")
            continue

        # 2. 等待 Clash 切换生效
        time.sleep(SWITCH_DELAY)

        # 3. 通过 Clash 混合端口测试连接
        print(f"正在通过 Clash ({CLASH_MIXED_PORT}) 测试 '{TARGET_URL}'...")
        if test_connectivity_via_clash(TARGET_URL, CLASH_MIXED_PORT):
            print(f"[✅ 成功] 节点 '{proxy_name}' 可以连接到 {TARGET_URL}。")
            successful_nodes.append(proxy_name)
        else:
            print(f"[❌ 失败] 节点 '{proxy_name}' 无法连接到 {TARGET_URL}。")

    print("\n" + "="*60)
    if successful_nodes:
        print(f"✅ **以下 {len(successful_nodes)} 个节点可以成功连接到 {TARGET_URL}:**")
        for node in successful_nodes:
            print(f"- {node}")
    else:
        print(f"❌ **没有节点可以成功连接到 {TARGET_URL}。**")
    print("="*60)

if __name__ == "__main__":
    main()