import importlib
import importlib.metadata
import platform
import sys

import torch


def check_python():
    print("\n--- Python Environment ---")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")

    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")

    print("[PASS] Python version")


def check_gpu():
    print("\n--- GPU / CUDA Verification ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")
        print("[INFO] CPU execution is supported.")

    print("[PASS] PyTorch installation")


def check_imports():
    print("\n--- Required Library Imports ---")

    packages = {
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "sentence-transformers": "sentence_transformers",
        "chromadb": "chromadb",
        "langgraph": "langgraph",
        "datasets": "datasets",
        "beautifulsoup4": "bs4",
        "langdetect": "langdetect",
        "pytest": "pytest",
    }

    failed = []

    for package, module_name in packages.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(package)
            print(f"[PASS] {package}: {version}")
        except Exception as error:
            print(f"[FAIL] {package}: {error}")
            failed.append(package)

    if failed:
        raise RuntimeError(
            f"Failed imports: {', '.join(failed)}"
        )


def main():
    print("=" * 55)
    print("AGENTIC RAG SYSTEM - ENVIRONMENT VERIFICATION")
    print("=" * 55)

    check_python()
    check_gpu()
    check_imports()

    print("\n" + "=" * 55)
    print("ENVIRONMENT VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
