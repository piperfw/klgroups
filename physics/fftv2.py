#!/usr/bin/env python

# Implements trapezoidal or cubic interpolation schemes with endpoint correction 

# TODO
# Example illustrating poor large-omega behaviour of simple FFT
# Example differentiating between trapezodial and cubic interpolations
# Example where asymptotic correction (currently commented out) may be useful

import numpy as np
from scipy.fft import ifft, fftfreq, fftshift

class dftint:
    """Implements trapezoidal and cubic interpolation schemes with endpoint
    corrections for Fourier integrals described by [1, Ch. 13]

    [1] W. H. Press et al., Numerical Recipes in C: The Art of Scientific Computing 2ed 
    (ISBN 0-521-43108-5), Cambridge University Press (1992).

    """
    # Threshold for series expansion of endpoint functions, error O(theta^8) [1, p. 696]
    # For very long sample times (large oversampling) it may be necessary to increase this
    SMALL_THETA = 1e-3 

    def __init__(self, times, upper_limit=None, FAC=4, order=4):
        """
        times: List (float) or np.ndarray (dtype('float64'))
            Array of equally spaced sample times of length M+1
        upper_limit: None or np.inf
            upper limit of integral, should be None (or np.inf) i.e. no
            upper limit, in which case an asymptotic correction may be made*
            [1, p. 699], or equal to times[-1]
        FAC: int or None
            Oversample factor for the DFT; the input array will be zero padded
            to the first integer power of 2 at least as large as FAC * (M+1)
            [1, p.694]. Pass FAC=None to prevent any oversampling.
            Recommended to oversample by at least a factor of 4 [1, p.695].
        order: 2 (Trapezoidal) or 4 (Cubic)
            Order of endpoint correction used as detailed in [1, Ch. 13].
            Trapezoidal has slightly less overhead, but may incur error 
            for functions sharply peaked in the frequency domain. 

        *code for the first correction is commented out below. I have not tested
        an example where this is useful to include yet, but if it might be in your
        application then hopefully it is a reasonable starting point (higher order
        terms of the expansion could be approximated using finite differences to
        get the derivatives of the function, see [1, Eq. 19.9.17]
        """
        
        assert np.all(np.isreal(times)), 'Parameter times must be a list or array of real values'
        self.a = times[0]
        self.dt = times[1] - times[0]
        assert np.allclose(np.diff(times), self.dt), 'Parameter times must be equally spaced'

        assert upper_limit in [None, np.inf, times[-1]],\
                f'Parameter upper_limit must be None (np.inf) or times[-1]={times[-1]}'
        self.b = times[-1] # upper limit of finite integral (set by times[-1] always)
        self.M = len(times)-1
            
        assert (type(FAC)==int and FAC > 0) or FAC is None,\
            'Parameter FAC must be a positive integer or None'
        self.N = self.M+1 if FAC is None else int(2**np.ceil(np.log2(FAC*self.M)))
        # Physical frequencies of result for this oversampling [+ve exponent, see fftfreq notes]
        self.omegas = 2 * np.pi * fftshift(fftfreq(self.N, d=self.dt))
        # Scaled frequencies used in computation, small value characterised by self.SMALL_THETA
        self.thetas = self.dt * self.omegas

        assert order in [2,4], 'Parameter order must be 2 or 4'
        # Compute correction functions at order (only needs to be done once)
        self.W, self.alphas = self._corr_funcs(order)
        self.alphas_conj = np.flip(np.conj(self.alphas), axis=0)

        self.asym_corr = 0.0
        # Asymptotic correction (unused)
        #if upper_limit in [None, np.inf]:
        #    self.asym_corr = self._asym_corr()

    def dftcorr(self, data):
        assert len(data) == self.M+1, f'length of data must match length of times={self.M+1}'
        # base DFT (ifft for +ve exponent; no normalisation; order ascending frequencies)
        dft0 = fftshift(ifft(data, norm='forward', n=self.N))
        # DFT with endpoint corrections, scaled by dt to give integral, [1, Eq. 13.9.13]
        # N.B. rows of self.alphas_conj are flipped relative to self.alphas so
        # data[-4] multiplies alpha^*_3, data[-3] alpha^*_2 etc. as in (13.9.13)
        dft1 = self.dt * np.exp(1j * self.omegas * self.a) * (\
                self.W * dft0 \
                + data[:4] @ self.alphas \
                + np.exp(1j * self.omegas * (self.b - self.a)) * (data[-4:] @ self.alphas_conj)
                )
        # asymptotic correction (possibly 0, see __init__)
        dft1 += self.asym_corr
        return dft1 

    def _asym_corr(self):
        large_omega = (2*np.pi)/(2*self.dt) # Nyquist freq. 
        i1 = next((i for i, omega in enumerate(self.omegas) if omega > - large_omega), 0)
        i2 = next((i for i, omega in enumerate(self.omegas) if omega >  large_omega), -1)
        omegas1, omegas2, omegas3 = self.omegas[:i1], self.omegas[i1:i2], self.omegas[i2:]
        return np.concatenate((
                    - np.exp(1j * omegas1 * self.b) / (1j * omegas1),
                    np.zeros(len(omegas2), dtype=complex),
                    - np.exp(1j * omegas3 * self.b) / (1j * omegas3)
                    ))

    def _corr_funcs(self, order):
        W = np.zeros(self.N, dtype=complex)
        alphas = np.zeros((4, self.N), dtype=complex) # 1 row for alpha0, alpha1,...
        if order == 2:
            for i, th in enumerate(self.thetas):
                if np.abs(th) < self.SMALL_THETA:
                    W[i] = 1 - (1/12)*th**2 + (1/360)*th**4 - (1/20160)*th**6
                    alphas[0,i] = (-1/2) + (1/24)*th**2 - (1/720)*th**4 + (1/40320)*th**6 + 1j*th*(1/6 - (1/120)*th**2 + (1/5040)*th**4 - (1/362880)*th**6)
                else:
                    W[i] = 2*(1-np.cos(th)) / th**2
                    alphas[0,i] = (-(1-np.cos(th))+1j*(th-np.sin(th)) ) / th**2
                    # alpha1, alpha2, alpha3 = 0 at this order
            return W, alphas
        # order==4 (cubic)
        for i, th in enumerate(self.thetas):
            if np.abs(th) < self.SMALL_THETA:
                W[i] = 1 - (11/720)*th**4 + (23/15120)*th**6
                alphas[0,i] = -(2/3)+(1/45)*th**2+(103/15120)*th**4-(169/226800)*th**6+1j*th*(2/45+(2/105)*th**2-(8/2835)*th**4+(86/467775)*th**6)
                alphas[1,i] = (7/24)-(7/180)*th**2+(5/3456)*th**4-(7/259200)*th**6 + 1j*th*(7/72-(1/168)*th**2+(11/72576)*th**4-(13/5987520)*th**6)
                alphas[2,i] = -(1/6)+(1/45)*th**2-(5/6048)*th**4+(1/64800)*th**6+1j*th*(-7/90+(1/210)*th**2-(11/90720)*th**4+(13/7484400)*th**6)
                alphas[3,i] = (1/24) - (1/180)*th**2 + (5/24192)*th**4-(1/259200)*th**6+1j*th*(7/360-(1/840)*th**2+(11/362880)*th**4-(13/29937600)*th**6)
            else:
                W[i] = ((6+th**2)/(3*th**4)) * ( 3 - 4 * np.cos(th) + np.cos(2*th))
                alphas[0,i] = (-42+5*th**2+(6+th**2)*(8*np.cos(th)-np.cos(2*th))+1j*(-12*th+6*th**3)+1j*(6+th**2)*np.sin(2*th))/(6*th**4)
                alphas[1,i] = (14*(3-th**2)-7*(6+th**2)*np.cos(th) + 1j*30*th - 5j*(6+th**2)*np.sin(th))/(6*th**4)
                alphas[2,i] = (-4*(3-th**2)+2*(6+th**2)*np.cos(th) + 1j*(-12*th) + 2j*(6+th**2)*np.sin(th))/(3*th**4)
                alphas[3,i] = (2*(3-th**2)-(6+th**2)*np.cos(th)+6j*th-1j*(6+th**2)*np.sin(th))/(6*th**4)
        return W, alphas

if __name__=='__main__':
    # FFT examples
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, figsize=(6,8))
    plt.subplots_adjust(hspace=0.3)

    w0 = 0.5
    Gamd = 0.015
    t0 = 0.0
    tf = 250.0
    dt = 0.5
    times = np.arange(t0, tf+dt, dt)

    # Function samples
    integrand_samples = np.cos(w0*times) * np.exp(-(Gamd/2)*times)
    # Objects to compute improved FFT at 2nd, 4th orders
    my_fft2 = dftint(times, upper_limit=None, FAC=16, order=2)
    my_fft4 = dftint(times, upper_limit=None, FAC=16, order=4)
    # Evaluation frequencies (same for both as identical FAC)
    omegas = my_fft2.omegas

    # Analytical result for transform, for comparison
    #exact = w0 / (w0**2 + (1j*omegas-0.5*Gamd)**2)
    #exact = 1/(0.5*Gamd-1j*(omegas-w0))
    exact = (Gamd/2 - 1j*omegas)/(w0**2+(Gamd/2-1j*omegas)**2)
    # Second order result
    result2 = my_fft2.dftcorr(integrand_samples)
    # Fourth order result
    result4 = my_fft4.dftcorr(integrand_samples)
    # Just DFT
    result0 = fftshift(ifft(integrand_samples, norm='forward', n=len(omegas)))

    # Time domain plot
    axes[0].plot(times, np.real(integrand_samples), label='Re f(t)')
    #axes[0].plot(times, np.imag(integrand_samples), label='Im')
    axes[0].set_xlabel(r't')
    axes[0].legend(title=r'cos(w0*t)exp(-(Gamd/2)t)')
    # Fourier domain plot
    axes[1].plot(omegas, np.real(exact), label='exact')
    axes[1].plot(omegas, np.real(result0), label='scipy.fft')
    axes[1].plot(omegas, np.real(result2), label='trapz.')
    axes[1].plot(omegas, np.real(result4), label='cubic', ls='--')
    axes[1].set_xlabel('w')
    axes[1].set_ylabel('Re', rotation=0)
    axes[1].legend()
    axes[1].set_xlim([0,1])
    axes[2].plot(omegas, np.imag(exact), label='exact')
    axes[2].plot(omegas, np.imag(result0), label='scipy.fft')
    axes[2].plot(omegas, np.imag(result2), label='trapz.')
    axes[2].plot(omegas, np.imag(result4), label='cubic', ls='--')
    axes[2].set_xlabel('w')
    axes[2].set_ylabel('Im', rotation=0)
    axes[2].set_xlim([0,1])
    # save
    fig.savefig('comparison.png', dpi=450, bbox_inches='tight')














