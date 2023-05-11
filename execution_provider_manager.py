import platform
import onnxruntime as onnx_rt
    
from openvino import utils
utils.add_openvino_libs_to_path()


class ExecutionProviderManager:
    def __init__(self, model, device):
        self.model = model
        self.session_option = onnx_rt.SessionOptions()
        self.device = device


    def get_inference_session(self):
        execution_provider = "CPUExecutionProvider"
        processor_name = platform.uname().processor
        execution_provider = "OpenVINOExecutionProvider" if "Intel" in processor_name else "DMLExecutionProvider"
        return onnx_rt.InferenceSession(self.model, self.session_option, providers =[execution_provider], provider_options=[{'device_type' : self.device}])