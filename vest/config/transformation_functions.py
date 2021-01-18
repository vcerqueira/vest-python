from vest.transformations.absolute import absolute
from vest.transformations.angle import vector_angle
from vest.transformations.cepstral_coefficients import mf_cepstral_coef
from vest.transformations.diff import diff1, diff2
from vest.transformations.dwt import DWT
from vest.transformations.ema import EMA
from vest.transformations.fourier import fft_imaginary, fft_real
from vest.transformations.noise_down import denoise_signal_wave
from vest.transformations.noise_up import noise_and_detrend
from vest.transformations.smd import SMD
from vest.transformations.sma import SMA
from vest.transformations.winsorisation import winsorise


TRANSFORMATIONS_FAST = \
    dict(absolute=absolute,
         diff1=diff1,
         diff2=diff2,
         SMA=SMA,
         fft_r=fft_real,
         noise=noise_and_detrend,
         SMD=SMD)

TRANSFORMATIONS_ALL = \
    dict(absolute=absolute,
         angle=vector_angle,
         cepstral=mf_cepstral_coef,
         diff1=diff1,
         diff2=diff2,
         DWT=DWT,
         EMA=EMA,
         SMA=SMA,
         fft_i=fft_imaginary,
         fft_r=fft_real,
         denoise=denoise_signal_wave,
         noise=noise_and_detrend,
         SMD=SMD,
         winsorise=winsorise
         )

N_PARAMETER = ['SMA', 'EMA', 'SMD']
