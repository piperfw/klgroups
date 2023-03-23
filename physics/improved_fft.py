#!/usr/bin/env python
import numpy as np

class Improvedfft:
    def W(self, th):
        return 1- (1/12)*th**2 + (1/360)*th**4 - (1/20160)*th**6
    def alpha0(self, th):
        return (-1/2) + (1/24)*th**2 - (1/720)*th**4 + (1/40320)*th**6 + 1j*th*(1/6 - (1/120)*th**2 + (1/5040)*th**4 - (1/362880)*th**6)
    def __init__(self, sample_times, over_sample_factor=4):
        self.dt = sample_times[1]-sample_times[0]
        self.a = sample_times[0]
        self.b = sample_times[-1]
        # Zero pad to at least over_sample_factor * times[-1]
        # Controls resolution in frequency (increase for higher resolution, reduce if have very many sample points)
        self.N = len(sample_times)-1
        # [optional] Choose a power of 2 as Cooley-Tukey algorithm most efficient in this case
        self.M = int(2**np.ceil( np.log2( over_sample_factor * self.N )))
        self.omegas = None
        self.Ws = None
        self.alpha0s = None
        self.prefactor = None
        self.scaled = True

    def fftfreq(self):
        # frequencies from -ve to +ve (fftshift as numpy gives odd ordering by default)
        if self.omegas is None:
            self.omegas = 2 * np.pi * np.fft.fftshift( np.fft.fftfreq( self.M, d=self.dt ))
        return self.omegas

    def fft(self, data, scaled=True):
        assert len(data) == self.N + 1, 'Data must be same length as sample times'
        if self.omegas is None:
            self.fftfreq()
        # interpolation and end_point corrections (compute once for given sample_times)
        if self.Ws is None:
            self.Ws = self.W( self.dt * self.omegas )
        if self.alpha0s is None:
            self.alpha0s = self.alpha0( self.dt * self.omegas )
        # prefactor used if scaled
        if self.prefactor is None:
            self.prefactor = self.dt * np.exp(1j * self.omegas * self.a)
        # normal fft - convention followed is positive exponent, no normalisation factor
        fft = np.fft.fftshift( np.fft.ifft( data, norm='forward', n=self.M ))
        # construct correction - in most cases right endpoint correction is negligible / unnecessary (normally b->infty, f(t)->0)
        fft_corrected = self.Ws * fft + self.alpha0s * data[0] \
                + np.exp(1j * self.omegas * (self.b - self.a)) * np.conjugate( self.alpha0s ) * data[self.N]
        if scaled:
            fft_corrected = self.prefactor * fft_corrected
        return fft_corrected


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Example use
    def f(t):
        return t**2 * np.exp(-t**2)
    #sample_times = np.linspace(0,40,num=50)
    dt = 1
    dt2 = dt/100
    sample_times = np.arange(start=0.0, stop=50, step=dt)
    sample_times2 = np.arange(start=0.0, stop=50, step=dt2)
    sample_data = f(sample_times)
    sample_data2 = f(sample_times2)
    N = 4 * len(sample_times)
    N2 = 4 * len(sample_times2)
    normal_fft =  dt * np.fft.fftshift(np.fft.ifft(sample_data, norm='forward', n=N))
    normal_omegas = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    normal_fft2 =  dt2 * np.fft.fftshift(np.fft.ifft(sample_data2, norm='forward', n=N2))
    normal_omegas2 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N2, d=dt2))
    myfft = Improvedfft(sample_times)
    corrected_fft = myfft.fft(sample_data)
    corrected_omegas = myfft.fftfreq()
    fig, axes = plt.subplots(2, figsize=(6,6))
    axes[0].plot(sample_times2, sample_data2, label='dt/100')
    axes[0].plot(sample_times, sample_data, label='dt', linestyle='--')
    axes[0].legend()
    axes[0].set_ylabel('f(t)')
    axes[1].plot(normal_omegas2, np.real(normal_fft2), label='normal (dt/100)')
    axes[1].plot(normal_omegas, np.real(normal_fft), label='normal', linestyle='--')
    axes[1].plot(corrected_omegas, np.real(corrected_fft), label='corrected', linestyle='--')
    axes[1].set_xlim([normal_omegas[0],normal_omegas[-1]])
    axes[1].legend()
    axes[1].set_ylabel('Re[fft]')
    plt.tight_layout()
    plt.savefig('fft_example.pdf')


