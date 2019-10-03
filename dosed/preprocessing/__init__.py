from .regularization import GaussianNoise, RescaleNormal, Invert
from .normalizers import clip, clip_and_normalize, mask_clip_and_normalize
from .frequency import spectrogram, get_interpolator
from .filters import get_bandpass, get_highpass, get_lowpass


dict_filters = {
    "clip": clip,
    "clip_and_normalize": clip_and_normalize,
    "mask_clip_and_normalize": mask_clip_and_normalize,
    "bandpass": get_bandpass,
    "highpass": get_highpass,
    "lowpass": get_lowpass,
    "spectrogram": spectrogram,
}


__all__ = [
    GaussianNoise,
    RescaleNormal,
    Invert,
    get_interpolator,
]
