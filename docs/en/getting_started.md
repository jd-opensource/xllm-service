# Compilation and Execution

## Container
First, download the image we provide:
```bash
docker pull xllm-ai/xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```
Then create the corresponding container:
```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host  --device=/dev/davinci0  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/queue_schedule:/var/queue_schedule -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /usr/local/sbin/:/usr/local/sbin/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /export/home:/export/home -w /export/home -v ~/.ssh:/root/.ssh  -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /home/:/home/  -v /runtime/:/runtime/  xllm-ai:xllm-0.6.0-dev-800I-A3-py3.11-openeuler24.03-lts-aarch64
```

## Compilation
```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

### etcd Installation
Use the installation script provided by etcd official:
```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

### Adding a Patch
`etcd_cpp_apiv3` depends on the cpprest static library, but cpprest is built as a dynamic library by default. Therefore, you need to add a patch to the CMakeLists.txt of cpprest:
```bash
bash prepare.sh
```

### xLLM Service Compilation
```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```
!!! warning "Possible Errors"
    Here may encounter installation errors about `boost-locale` and `boost-interprocess`: `vcpkg-src/packages/boost-locale_x64-linux/include: No such file or directory`,`/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`
    We use `vcpkg` to reinstall these packages:
    ```bash
    /path/to/vcpkg remove boost-locale boost-interprocess
    /path/to/vcpkg install boost-locale:x64-linux
    /path/to/vcpkg install boost-interprocess:x64-linux
    ```

## Execution
1. First, start the etcd service:
```bash 
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

2. Then start the xllm-service service:
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=0 \
ENABLE_XLLM_DEBUG_LOG=1 \
./build/xllm_service/xllm_master_serving \
    --etcd_addr="127.0.0.1:2389" \
    --http_server_port=9888 \
    --rpc_server_port=9889 \
    --tokenizer_path /path/to/tokenizer_config/
```

xllm-service needs to start an http service and an rpc service. The http service is used to receive and process user requests, and the rpc service is used to interact with xllm instances.

The complete usage process needs to be used with xllm, please refer to the link: [xLLM PD Disaggregated Deployment](https://xllm.readthedocs.io/zh-cn/latest/zh/getting_started/PD_disagg/)

### service Parameters
http service：It is used to receive and process user requests.
| Parameter | Description | Default Value |
| --- | --- | --- |
| http_server_host | http service address | "" |
| http_server_port | http service port | 8888 |
| http_server_idle_timeout_s | http service timeout | -1 |
| http_server_num_threads | http service thread number | 32 |
| http_server_max_concurrency | http service max concurrency | 128 |

rpc service：It is used to interact with xllm, manage the status of xllm instance clusters, etc.
| Parameter | Description | Default Value |
| --- | --- | --- |
| rpc_server_host | rpc service address | "" |
| rpc_server_port | rpc service port | 8889 |
| rpc_server_idle_timeout_s | rpc service timeout | -1 |
| rpc_server_num_threads | rpc service thread number | 32 |
| rpc_server_max_concurrency | rpc service max concurrency | 128 |

Environment Variables:
ENABLE_DECODE_RESPONSE_TO_SERVICE: In the PD disaggregated scenario, whether to return the decoding result to the service directly(without forwarding through the P instance), 0 means "no", 1 means "yes".
ENABLE_XLLM_DEBUG_LOG: Whether to enable xllm debug log, 0 means "no", 1 means "yes".
