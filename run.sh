echo "Running the compile script..."
bash compile_run.sh

echo "Simplifying the model..."
python -m onnxsim model.onnx model_simplified.onnx

echo "Running the eval script..."
bash eval_run.sh