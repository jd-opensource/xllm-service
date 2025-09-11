# 编译与运行

## 容器
首先下载我们提供的镜像：
```bash
docker pull xllm-ai/xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```
然后创建对应的容器
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm-ai:xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```

## 编译
```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

### etcd安装
使用etcd官方提供的[安装脚本](https://github.com/etcd-io/etcd/releases)进行安装，其脚本提供的默认安装路径是`/tmp/etcd-download-test/etcd`，我们可以手动修改其脚本中的安装路径，也可以运行完脚本之后手动迁移：
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

### 添加补丁
etcd_cpp_apiv3 依赖 cpprest 静态库，但 cpprest 编译产生的是动态库，因此需要给 cpprest 的 CMakeLists.txt 加一个补丁：
```bash
bash prepare.sh
```

### xLLM Service编译
再执行编译:
```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```
!!! warning "可能的错误"
    这里能会遇到关于`boost-locale`和`boost-interprocess`的安装错误：`vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    我们使用`vcpkg`重新安装这些包:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

## 运行
1. 首先需要启动etcd服务:
```bash 
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

2. 然后启动service服务:
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=0 \
ENABLE_XLLM_DEBUG_LOG=1 \
./build/xllm_service/xllm_master_serving \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889 \
    --tokenizer_path /path/to/tokenizer_config/
```

xllm-service需要启动一个http服务和一个rpc服务，http服务用于对外接收与处理用户请求，rpc服务用于和xllm实例进行交互。

完整的使用流程需要结合xllm一起使用，请查看链接: [xLLM PD分离部署](https://xllm.readthedocs.io/zh-cn/latest/zh/getting_started/PD_disagg/)

### service参数
http服务：用于对外接收以及处理用户请求。
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| http_server_host | http 服务地址 | "" |
| http_server_port | http 服务端口 | 8888 |
| http_server_idle_timeout_s | http 服务超时时间 | -1 |
| http_server_num_threads | http 服务线程数 | 32 |
| http_server_max_concurrency | http 服务最大请求并发数 | 128 |

rpc服务：用于与xllm之间交互，管理xllm实例集群状态等。
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| rpc_server_host | rpc 服务地址 | "" |
| rpc_server_port | rpc 服务端口 | 8889 |
| rpc_server_idle_timeout_s | rpc 服务超时时间 | -1 |
| rpc_server_num_threads | rpc 服务线程数 | 32 |
| rpc_server_max_concurrency | rpc 服务最大请求并发数 | 128 |

环境参数:
ENABLE_DECODE_RESPONSE_TO_SERVICE: 在PD分离场景下，是否将解码结果直接返回给service(不需要经过P实例转发)，0表示“否”，1表示“是”。
ENABLE_XLLM_DEBUG_LOG: 是否开启xllm debug log，0表示不开启，1表示开启。
