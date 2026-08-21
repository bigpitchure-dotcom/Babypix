"""
Convert merged model to GGUF format for Ollama.

Usage:
    python scripts/convert_to_gguf.py
"""

import os
import subprocess
import sys

MERGED_PATH = "./babypix1_coding_merged"
GGUF_PATH = "./babypix1_coding.gguf"
LLAMA_CPP_PATH = "./llama.cpp"
CONVERT_SCRIPT = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")


def check_llama_cpp():
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: {CONVERT_SCRIPT} not found")
        print("Run: git clone https://github.com/ggerganov/llama.cpp.git")
        sys.exit(1)


def convert_to_gguf():
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        MERGED_PATH,
        "--outfile",
        GGUF_PATH,
        "--outtype",
        "q8_0",
    ]

    print(f"Converting {MERGED_PATH} to GGUF...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(f"GGUF model saved to: {GGUF_PATH}")
    print(f"File size: {os.path.getsize(GGUF_PATH) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    check_llama_cpp()
    convert_to_gguf()
