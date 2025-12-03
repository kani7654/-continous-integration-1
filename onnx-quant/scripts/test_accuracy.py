import onnxruntime as ort
import numpy as np

def infer(model_path, inputs):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    preds = []
    for x in inputs:
        out = sess.run(None, {input_name: x})[0]
        pred = np.argmax(out, axis=1)
        preds.append(pred)
    return np.concatenate(preds)

if __name__ == "__main__":
    np.random.seed(0)
    inputs = [np.random.rand(1, 1, 28, 28).astype(np.float32) for _ in range(10)]

    fp_preds = infer("model.onnx", inputs)
    int_preds = infer("model_int8.onnx", inputs)

    agree = (fp_preds == int_preds).mean()
    print(f"FP32 vs INT8 agreement: {agree * 100:.2f}%")

    if agree < 0.85:
        print("Agreement below threshold — failing")
        raise SystemExit(1)
    else:
        print("Agreement OK")
