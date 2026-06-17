"""Models facade module.

This module exposes model classes used by the GUI/simulator. Currently the
concrete implementations live inside `train_adjusted_sharpe.py`. We re-export
them here so migration can happen progressively.
"""
try:
    import train_adjusted_sharpe as _tas
except Exception:
    _tas = None

if _tas is not None and hasattr(_tas, 'SimpleLSTMPolicy') and hasattr(_tas, 'SimpleMLP'):
    SimpleLSTMPolicy = _tas.SimpleLSTMPolicy
    SimpleMLP = _tas.SimpleMLP
else:
    class SimpleLSTMPolicy:
        def __init__(self, *args, **kwargs):
            raise ImportError('SimpleLSTMPolicy is not available: please ensure train_adjusted_sharpe.py is importable')

    class SimpleMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError('SimpleMLP is not available: please ensure train_adjusted_sharpe.py is importable')

__all__ = ['SimpleLSTMPolicy', 'SimpleMLP']
