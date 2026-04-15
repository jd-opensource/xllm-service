# Release xllm-service 0.9.1

## **Major Features and Improvements**

### Feature

- Support etcd auth in etcd client via environment variables.
- Support namespace prefix for etcd key.
- Implement instance readiness gate for HTTP service.


# Release xllm-service 0.9.0

## **Major Features and Improvements**

### Feature

- Support disaggregated prefill and decoding.
- Support KV Cache aware routing.
- Support KV Cache Pool.
- Support instance incarnation tracking and lease-lost failover.
- Support tool call and reasoning parser in PD disagg mode.
- Support chat_template_kwargs in chat completions.
- Support service to respond cancel status to instance upon client disconnection.
- Optimize kv cache hash algorithm and improve performance.
- Replace all http with rpc for interaction with xLLM instance.
- Support multi xllm_service sending request to single xllm instance.
- Add bvar to monitor xllm service metrics.

### Bugfix
- Fix incorrect behavior when prefill instance shutdown and first token not form str.
- Fix hang issue caused by unreleased requests after instance shutdown.
- Set correct parameters for client output.