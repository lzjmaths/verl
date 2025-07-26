export http_proxy="http://127.0.0.1:17890"
export https_proxy="http://127.0.0.1:17890"
export all_proxy="socks5://127.0.0.1:17890"
export no_proxy="localhost,127.0.0.1,::1"

export http_proxy="http://127.0.0.1:10090"
export https_proxy="http://127.0.0.1:10090"
export all_proxy="socks5://127.0.0.1:10090"
export no_proxy="localhost,127.0.0.1,::1"

unset http_proxy
unset https_proxy
unset all_proxy
unset no_proxy

curl -X PUT "http://127.0.0.1:10090/proxies/%E2%99%BB%EF%B8%8F%20%E8%87%AA%E5%8A%A8%E9%80%89%E6%8B%A9" -d '{"name":"16 | 韩国 | 1000M带宽 | 外贸专线C"}' -H "Content-Type: application/json"