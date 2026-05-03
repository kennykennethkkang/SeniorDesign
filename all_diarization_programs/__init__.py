__all__ = ["NemoMsddDiarizer", "PyannoteCallhomeDiarizer"]


def __getattr__(name):
    if name == "NemoMsddDiarizer":
        from .nemo_msdd_backend import NemoMsddDiarizer

        return NemoMsddDiarizer
    if name == "PyannoteCallhomeDiarizer":
        from .pyannote_callhome_backend import PyannoteCallhomeDiarizer

        return PyannoteCallhomeDiarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
