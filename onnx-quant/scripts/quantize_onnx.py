from onnxruntime.quantization import quantize_dynamic, QuantType
import sys

FP32 = "model.onnx"
INT8 = "model_int8.onnx"

if __name__ == "__main__":
    try:
        quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
        print(f"Wrote quantized model to {INT8}")
    except Exception as e:
        print("Quantization failed:", e)
        sys.exit(1)
