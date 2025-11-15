# Third‑Party Notices

This distribution includes third‑party software components. The notices below are provided for your convenience to comply with applicable license terms.

No model weights are included in this package. Models must be pulled by the end user and are governed by their own licenses.

---

## 1) Ollama (MIT)

- Project: https://github.com/ollama/ollama  
- License: MIT  
- Copyright: © Ollama  
- License Text: Included in this distribution at:
  - `blender_assistant_mcp/bin/LICENSE.ollama`
- Upstream License (online):  
  https://raw.githubusercontent.com/ollama/ollama/main/LICENSE

This distribution may include `ollama.exe` and related support files to run a local HTTP server used by the add‑on. Redistribution of Ollama binaries is subject to the MIT license above.

---

## 2) llama.cpp and ggml (MIT)

Ollama internally uses components from the following projects (among others), which are licensed under MIT:

- llama.cpp  
  - Project: https://github.com/ggerganov/llama.cpp  
  - License (MIT): https://raw.githubusercontent.com/ggerganov/llama.cpp/master/LICENSE

- ggml  
  - Project: https://github.com/ggerganov/ggml  
  - License (MIT): https://raw.githubusercontent.com/ggerganov/ggml/master/LICENSE

Where these components are bundled within the Ollama-provided binaries, they remain governed by their respective MIT licenses.

---

## 3) NVIDIA CUDA Runtime Components (CUDA Toolkit EULA)

If this package includes NVIDIA CUDA runtime redistributables (for example):

- `cudart64_12.dll`
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- Other CUDA/cuBLAS/cuDNN DLLs as applicable

then their use and redistribution are governed by the NVIDIA CUDA Toolkit End User License Agreement (EULA):

- NVIDIA CUDA Toolkit EULA:  
  https://docs.nvidia.com/cuda/eula/index.html

You are responsible for ensuring your use and redistribution of these NVIDIA components complies with the CUDA Toolkit EULA and any applicable NVIDIA terms.

---

## 4) Models Not Included

No LLM or VLM model weights are shipped with this package. Users may “pull” models at runtime (e.g., via Ollama). Each model has its own license and restrictions. Please review and comply with the license terms of any model you download and use.

---

## 5) Source and License Availability

- Ollama (MIT):  
  Source: https://github.com/ollama/ollama  
  License: `blender_assistant_mcp/bin/LICENSE.ollama` (included) and online at the link above.

- llama.cpp (MIT):  
  Source: https://github.com/ggerganov/llama.cpp  
  License: https://raw.githubusercontent.com/ggerganov/llama.cpp/master/LICENSE

- ggml (MIT):  
  Source: https://github.com/ggerganov/ggml  
  License: https://raw.githubusercontent.com/ggerganov/ggml/master/LICENSE

- NVIDIA CUDA runtime (EULA):  
  EULA: https://docs.nvidia.com/cuda/eula/index.html

If you believe an attribution or license notice is missing or incomplete, please open an issue so we can correct it.

---

This notices file is provided for informational purposes only and does not modify or supersede the actual licenses that govern the third‑party components.