<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)

## 1. Project Overview
**xLLM-service** is a service-layer framework developed based on the **xLLM** inference engine, providing efficient, fault-tolerant, and flexible LLM inference services for clustered deployment.

xLLM-service targets to address key challenges in enterprise-level service scenarios:

- How to ensure the SLA of online services and improve resource utilization of offline tasks in a hybrid online-offline deployment environment.

- How to react to changing request loads in actual businesses, such as fluctuations in input/output lengths.

- Resolving performance bottlenecks of multimodal model requests.

- Ensuring high reliability of computing instances.

---

## 2. Key Features
With management of computing resource pools, intelligent scheduling and preemption of hybrid requests, and real-time monitoring of computing instances, xLLM-service achieves the following key features:

- Unified scheduling of online and offline requests, with preemptive execution for online requests and best-effort execution for offline requests.

- Adaptive dynamic allocation of PD ratios, supporting efficient switching of instance PD roles.

- EPD three-stage disaggregation for multimodal requests, with intelligent resource allocation for different stages.

- Fault-tolerant architecture, fast detection of instance error and automatic rescheduling for interrupted requests. 

---

## 3. Core Architecture

```
├── xllm-service/
|   : main source folder
│   ├── chat_template/               # 
│   ├── common/                      # 
│   ├── examples/                    # 
│   ├── http_service/                # 
│   ├── rpc_service/                 # 
|   ├── tokenizers/                  #
|   └── master.cpp                   # 
```

---


## 4. Quick Start
#### Installation
```
git clone git@coding.jd.com:xllm-ai/xllm_service.git
cd xllm_service
git submodule init
git submodule update
```
#### Compilation
compile vcpkg, set env variable:
```
export VCPKG_ROOT=/export/home/xxx/vcpkg-src
```
compile xllm-service: 
```
mkdir -p build && cd build
cmake .. && make -j 8
```


--- 

## 5. Contributing

There are several ways you can contribute to xLLM:

1. Reporting Issues (Bugs & Errors)
2. Suggesting Enhancements
3. Improving Documentation
    + Fork the repository
    + Add your view in document
    + Send your pull request
4. Writing Code
    + Fork the repository
    + Create a new branch
    + Add your feature or improvement
    + Send your pull request

We appreciate all kinds of contributions! 🎉🎉🎉
If you have problems about development, please check our document: * **[Document](./docs/docs/readme.md)**

---

## 6. Community & Support

If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack.

Welcome to contact us:

<div align="center">
  <img src="" alt="contact" width="50%" height="50%">
</div>

---
## 7. About the Contributors

Thanks to all the following [developers](https://github.com/jd-opensource/xllm-service/graphs/contributors) who have contributed to xLLM.
<a href="https://github.com/jd-opensource/xllm-service/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm-service" />
</a>

---

## 8. License

[Apache License](LICENSE)

#### xLLM is provided by JD.com 
#### Thanks for your Contributions!
