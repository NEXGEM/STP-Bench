from .contrastive import ContrastiveAdapter
from .default import DefaultAdapter
from .egn import EGNAdapter
from .graph import GraphAdapter
from .sepal import SepalAdapter
from .stem import StemAdapter
from .triplex import TriplexAdapter


_ADAPTER_TYPES = {
    "default": DefaultAdapter,
    "egn": EGNAdapter,
    "triplex": TriplexAdapter,
    "graph": GraphAdapter,
    "contrastive": ContrastiveAdapter,
    "sepal": SepalAdapter,
    "stem": StemAdapter,
}
_CLASS_FALLBACKS = {}


def register_adapter(name, adapter_cls):
    _ADAPTER_TYPES[name] = adapter_cls


def register_class_fallback(class_name, adapter_cls):
    _CLASS_FALLBACKS[class_name] = adapter_cls


def get_adapter(class_name=None, adapter_name=None):
    if adapter_name:
        if adapter_name not in _ADAPTER_TYPES:
            raise ValueError(f"Unknown model adapter: {adapter_name}")
        return _ADAPTER_TYPES[adapter_name]()
    return _CLASS_FALLBACKS.get(class_name, DefaultAdapter)()
