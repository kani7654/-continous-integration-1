# ONNX Model Conversion CI Pipeline

This project automates the conversion and validation of a PyTorch model using **GitHub Actions**.

##  Goals
- Export PyTorch model → ONNX format
- Perform a mock INT8 quantization step
- Validate that model accuracy is not affected
- Run everything automatically on each commit

---

##  CI Workflow Steps

When new code is pushed to GitHub:

1️ Install dependencies  
2️ Export PyTorch CNN to **model.onnx**  
3️ Mock quantize → create **model_int8.onnx**  
4️ Run output agreement test (FP32 vs INT8)  
5️ Upload ONNX models as CI artifacts  

If accuracy drops → CI fails   
If both models behave correctly → CI passes 

---


