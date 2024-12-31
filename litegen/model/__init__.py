# from .main import lazy_completion as completion

from .completions import (
    lazy_completion as completion,
    gpu_lazy_completion as gpu_completion,
    print_stream_completion as pp_completion
)