SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" &&
echo "=== exporting YoloV8n in FP32 and FP16 ===" &&
python3 "$SCRIPT_DIR/../python/export_yolo8_onnx.py" &&


echo "=== Building TensorRT FP32 engine ===" &&

trtexec \
    --onnx="../models/yolov8n_fp32.onnx" \
    --saveEngine="../engines/yolov8n_fp32.engine" \
    --noTF32 &&

echo "=== Building TensorRT FP16 engine ===" &&

trtexec \
    --onnx="../models/yolov8n_fp16.onnx" \
    --saveEngine="../engines/yolov8n_fp16.engine" \
    --noTF32 &&

echo "=== Engine build completed successfully ==="

