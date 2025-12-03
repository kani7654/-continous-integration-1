

import shutil
import os
import sys

FP32 = "model.onnx"
INT8 = "model_int8.onnx"

if __name__ == "__main__":
    if not os.path.exists(FP32):
        print(f"Source model not found: {FP32}")
        sys.exit(1)

    shutil.copy(FP32, INT8)
    print(f"Mock quantization: copied {FP32} -> {INT8}")
