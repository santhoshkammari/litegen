from litegen.model._oai import OmniLLMClient


class ServingModel:
    def __init__(self):
        self.gpu_flag = None
        self._cpu_client = None
        self._gpu_client = None

    @property
    def completion(self):
        self._update_client(gpu=False)
        return getattr(self, self._get_client_attr(gpu=False)).completion

    @property
    def completion_gpu(self):
        self._update_client(gpu=True)
        return getattr(self, self._get_client_attr(gpu=True)).completion

    def _get_client_attr(self, gpu):
        return '_gpu_client' if gpu else '_cpu_client'

    def _update_client(self, gpu):
        client_attr = self._get_client_attr(gpu)
        if getattr(self, client_attr) is None:
            setattr(self, client_attr, OmniLLMClient(gpu=gpu))

    def _get_partial_fun(self, model: str, gpu: bool = False):
        """Creates a partial function that handles both message string and complex inputs."""
        self._update_client(gpu)
        client = getattr(self, self._get_client_attr(gpu))

        def wrapper(*args, **kwargs):
            # Remove gpu from kwargs before passing to completion
            kwargs.pop('gpu', None)

            if len(args) == 1 and isinstance(args[0], str):
                # Handle simple string input
                return client.completion(messages = args[0],model=model, **kwargs)
            else:
                # Handle other input types
                return client.completion(*args,model=model, **kwargs)

        return wrapper

    def _create_model_property(model_name:str):
        """Factory function to create model properties"""

        def model_property(self):
            return lambda *args, **kwargs: self._get_partial_fun(
                model=model_name,
                gpu=kwargs.get('gpu', False)
            )(*args, **kwargs)

        return property(model_property)

    # Define model properties using the factory function
    llama3p2_3b_instruct_q4_K_M = _create_model_property("llama3.2:3b-instruct-q4_K_M")
    llama3p2_1b_instruct_q4_K_M = _create_model_property("llama3.2:1b-instruct-q4_K_M")

    qwen2p5_0p5b_instruct = _create_model_property("qwen2.5:0.5b-instruct")
    qwen2p5_3b_instruct = _create_model_property("qwen2.5:3b-instruct")
    qwen2p5_7b_instruct = _create_model_property("qwen2.5:7b-instruct")

    qwen2p5_coder_1p5b_instruct = _create_model_property("qwen2.5-coder:1.5b-instruct")
    qwen2p5_coder_0p5b_instruct = _create_model_property("qwen2.5-coder:0.5b-instruct")
    qwen2p5_coder_7b_instruct = _create_model_property("qwen2.5-coder:7b-instruct")


if __name__ == '__main__':
    model = ServingModel()
    _BASE_MODEL= 'qwen2.5:0.5b-instruct'
    # Test CPU usage
    print(model.qwen2p5_p5b('what is 2+3?'))
    # Test GPU usage
    print(model.qwen2p5_p5b('what is 2+3?', gpu=True))

    print(model.completion(model=_BASE_MODEL,
                           messages = "hi"))
