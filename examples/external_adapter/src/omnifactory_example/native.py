"""Replace these methods for a new architecture; engine code stays unchanged."""

from omnifactory.adapters.qwen2_5 import Qwen25Adapter


class ExampleAdapter(Qwen25Adapter):
    """Runs the random tiny native Qwen fixture, not a supported pretrained model."""
