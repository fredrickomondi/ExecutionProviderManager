import platform
from execution_provider_manager import ExecutionProviderManager


if platform.system() == "Windows":
    from openvino import utils
    utils.add_openvino_libs_to_path()


model = "my_model.onnx"
ep_manager = ExecutionProviderManager(model, "GPU_FP16") # assumes GPU is the inference device, for more device flags check https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
inference_session = ep_manager.get_inference_session()
#inference_session.run()
