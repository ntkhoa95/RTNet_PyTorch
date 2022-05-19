import torch, os, onnxruntime, onnx
from models.RTFNet import RTFNet
from utils.util import *

device = torch.device('cpu')
model = RTFNet(n_class=3, num_resnet_layers=18, verbose=False)
model_checkpoint_dir = os.path.join("./checkpoints", "gmrpd_ALSDL", "RTFNet18")
load_network(model, "400", model_checkpoint_dir)
model = model.to(device)
model.eval()

rgb = torch.randn(1, 3, 480, 640).to(device)
thermal = torch.randn(1, 1, 480, 640).to(device)
tensor_inputs = torch.cat((rgb, thermal), dim=1)

rtf_net = RTFNet(3).to(device)
output = rtf_net(tensor_inputs)
print(output.shape)

output_file = "rtfnet.onnx"

with torch.no_grad():
    torch.onnx.export(
        model, tensor_inputs,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
    print(f'Successfully exported ONNX model: {output_file}')

onnx_model = onnx.load(output_file)
if onnx_model.ir_version < 4:
    print("Model with ir_version below 4 requires to include initilizer in graph input")

inputs = onnx_model.graph.input
name_to_input = {}
for input in inputs:
    name_to_input[input.name] = input

for initializer in onnx_model.graph.initializer:
    if initializer.name in name_to_input:
        inputs.remove(name_to_input[initializer.name])

onnx.save(onnx_model, output_file)

onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(output_file)
# compute ONNX Runtime output prediction
ort_outs = ort_session.run(None, {'input': tensor_inputs.numpy()})
print(ort_outs)