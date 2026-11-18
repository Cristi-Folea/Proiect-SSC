echo "=== Building TensorRT FP32 engine ==="

trtexec \
    --onnx="../models/yolov8n.onnx" \
    --saveEngine="../engines/yolov8n_fp32.engine" \
    --noTF32

echo "=== Building TensorRT FP16 engine ==="

trtexec \
    --onnx="../models/yolov8n_fp16.onnx" \
    --saveEngine="../engines/yolov8n_fp16.engine" \
    --noTF32

echo "=== Engine build completed successfully ==="

