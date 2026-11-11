echo "=== Building TensorRT FP32 engine ==="

trtexec \
    --onnx="$MODEL_DIR/yolov8n.onnx" \
    --saveEngine="$ENGINE_DIR/yolov8n_fp32.engine" \
    --noTF32

echo "=== Building TensorRT FP16 engine ==="

trtexec \
    --onnx="$MODEL_DIR/yolov8n_fp16.onnx" \
    --saveEngine="$ENGINE_DIR/yolov8n_fp16.engine" \
    --noTF32

echo "=== Engine build completed successfully ==="