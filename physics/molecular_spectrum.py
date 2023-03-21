#!/usr/bin/env python
import argparse, os, sys, pickle
import numpy as np
from time import time
from scipy.signal import argrelextrema
from scipy.constants import c, hbar, e, pi, k
from scipy import special
from mpmath import fp
import scipy.integrate as integrate
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
# used for calls to scipy.integrate.quad
MAX_SUBINTERVALS=1000
MAX_CYCLES=100
# Output directory for plots
FIG_DIR = 'figures'
DATA_DIR = 'data'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

parser = argparse.ArgumentParser(description='Calculate single molecule absorption spectra. Use -plot FILE[S] to plot previously calculated data.')
parser.add_argument('-plot', nargs='+', metavar='FILE[S]', help='Plot previously calculated data (.npy files)')
group_env = parser.add_argument_group('Phonon environment')
group_sys = parser.add_argument_group('System')
group_comp = parser.add_argument_group('Computational')
group_plot = parser.add_argument_group('Plotting')
group_env.add_argument('-s', default=1, type=float, help='Spectral density Ohmicity')
group_env.add_argument('-nuc', default=0.15, type=float, help='Spectral density cut-off frequency (eV)')
group_env.add_argument('-a', default=0.25, type=float, help='System-bath coupling strength')
group_env.add_argument('-T', default=0.026, type=float, help='Phonon temperature (eV)')
group_env.add_argument('-TK', default=None, type=float, help='Phonon temperature (K) [overrides -T]')
group_sys.add_argument('-w0', default=2.31, type=float, help='Electronic transition frequency (eV)')
group_sys.add_argument('-dissipation', default=0.01, type=float, help='Incoherent loss rate (eV)')
group_sys.add_argument('-drive', default=0.00, type=float, help='Incoherent gain rate (eV) [UNTESTED]')
group_sys.add_argument('-kappa', type=float, default=0.01, help='Cavity loss rate (eV)')
group_sys.add_argument('-gn', type=float, default=0.1, help='Collective light-matter coupling (eV)')
group_comp.add_argument('-dt', default=0.05, type=float, help='Timestep to sample integrand (correlator)')
group_comp.add_argument('-st', default=300, type=int, help='Final sample time')
group_comp.add_argument('--correct-fft', action='store_true', help='Use corrected FFT algorithm')
group_comp.add_argument('-over-sample-factor', default=16, type=int, help='Pad data up to this may times it\'s original length')
group_plot.add_argument('-xlims', nargs=2, type=float, default=[-1000,1000], help='Min Max detuning (meV) e.g. -1000 1000 or wavelength (nm) to plot')
group_plot.add_argument('--no-normalise', action='store_true', help='Don\'t normalise the spectrum')
group_plot.add_argument('--emission', action='store_true', help='Plot emission predicted by Kennard-Stepanov relation')
group_plot.add_argument('--wavelengths', action='store_true', help='Plot wavelength (nm) on x-axis instead of frequency (eV)')
group_plot.add_argument('--no-w0', action='store_true', help='Don\'t plot zero phonon line')
group_plot.add_argument('-png', action='store_true', help='Save plots as .png (rahter than .pdf)')
args=vars(parser.parse_args())
EV_TO_HZ=(e/hbar) # convert from eV to Hz
HZ_TO_NM=1e9*(2*pi*c) # convert from angular freq. (Hz) to wavelength (nm.)
K_TO_EV = (k/e) # convert from Kelvin to eV
# Get temperature in eV
if args['TK'] is not None:
    args['T'] = K_TO_EV * args['TK']
else:
    args['TK'] = round(args['T'] / K_TO_EV, 0)
# plot vs frequencies unless wavelengths specified
if args['wavelengths']:
    XLABEL=r'\(\(\lambda\) \rm{(nm)}\)' 
    XSCALE=1
    XLIMS = args['xlims']
else:
    XLABEL=r'\(\omega-\omega_0\) \rm{(meV)}'
    XSCALE=1000 # plot meV instead of eV
    args['freqs'] = True
    XLIMS = args['xlims']
# Setup matplotlib params
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', **{'size':18})
linewidth=2.5
save_ext='.png' if args['png'] else '.pdf' 
#YTICKS=[0.0,0.5,1.0] if not args['no_normalise'] else None
YTICKS = None


# Conveninent varaibles
nuc=args['nuc'] # bath cutoff freq
a=args['a']     # system-bath coupling
w0=args['w0']   # 2LS frequency
lambda0 = (1/EV_TO_HZ)*(1/w0)*HZ_TO_NM # corresponding wavelength
kappa = args['kappa']
gn = args['gn']
T=args['T']     # Phonon bath temperature in eV
TK=T/K_TO_EV    # Kelvin
gamma_down=args['dissipation'] # 2LS dissipation
gamma_up=args['drive'] # 2LS drive
gamma_T=gamma_down+gamma_up
s=args['s']     # Ohmicity of bath
dt=args['dt']   # Timestep between samples
st=args['st']   # Total sample time
PLOT_W0 = not args['no_w0']
tcheckpoints=[st//4, 2*st//4, 3*st//4] # provide progress updates (stdout) at 25, 50, 75%
t0=time()
# improved fft
def fft_corr(times, data, over_sample_factor=16, data_fval=None):
        def W(th):
            return 1- (1/12)*th**2 + (1/360)*th**4 - (1/20160)*th**6
        def alpha0(th):
            return (-1/2) + (1/24)*th**2 - (1/720)*th**4 + (1/40320)*th**6 + 1j*th*(1/6 - (1/120)*th**2 + (1/5040)*th**4 - (1/362880)*th**6)
        M = len(times)-1
        dt = times[1]-times[0]
        N = over_sample_factor * M
        fft = np.fft.fftshift(np.fft.ifft(data, norm='forward', n=N))
        # endpoints (independent var)
        a = times[0]
        b = times[-1]
        nus = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=dt))
        plot_nus = nus + w0
        thetas = dt * nus
        ws = W(thetas)
        alpha0s = alpha0(thetas)
        # apply corrections to fft
        fft_corr = np.array([])
        fval = data_fval if data_fval is not None else data[M]
        for i,omega in enumerate(nus):
            corr = np.exp(1j*a*omega)*(ws[i] * fft[i] + alpha0s[i] * data[0] +
                np.exp(1j*(b-a)*omega)*np.conjugate(alpha0s[i])*fval)
            fft_corr = np.append(fft_corr, corr)
        return plot_nus, fft_corr
# Spectral density (defined for positive freq.)
def J(nu):
    return 2*a*nu**s*(1/nuc)**(s-1)*np.exp(-(nu/nuc)**2)
# Integrand of correlation function split up into Re and Im parts
def CpRe(nu, t):
    if T == 0.0:
        return (J(nu)/nu**2)*(1-np.cos(t*nu))
    return (J(nu)/nu**2)*((1/np.tanh(nu/(2*T)))*(1-np.cos(t*nu)))
def CpIm(nu, t):
    return (J(nu)/nu**2)*np.sin(t*nu)
def C(t):
    if T==0.0 and s==1:
        # closed form expression in this case [result, 'error']
        re_int = [0.5 * a * t**2 * nuc**2 * fp.hyp2f2(1,1,1.5,2, -0.25 * t**2 * nuc**2), 0.0]
    else:
        re_int = integrate.quad(CpRe, 0, np.inf, args=(t), limit=MAX_SUBINTERVALS)
    # perform weighted integral - this is normally more performant but I have commented it out at some point (?)
    #im_int = integrate.quad(CpIm, 0, np.inf, args=(t), limit=MAX_SUBINTERVALS, weight='sin', wvar=t, limlst=MAX_CYCLES)
    if s == 1:
        # Ohmic bath has closed form expression
        im_int = [a * pi * special.erf(t * nuc / 2), 0.0]
    else:
        im_int = integrate.quad(CpIm, 0, np.inf, args=(t), limit=MAX_SUBINTERVALS)
    # (result, error)
    # Error discarded here
    return re_int[0] + 1j*im_int[0]
# Integrand
def f(t):
    if len(tcheckpoints) > 0 and t > tcheckpoints[0]:
        print('At t={:.0f} (runtime = {:.0f}s)'.format(t, time()-t0))
        del tcheckpoints[0]
    # 2021-06-25: Fixed factor of (1/2) error here (cf. Kirton and Keeling 2015 eq. 4. & 5.)
    # 2021-10-22: Now multiply by gamma_T/2 t afterwards, so can export just correlator for use in weak coupling
    return np.exp(-C(t))
    #return np.exp( -gamma_T*t/2 -C(t))
# Plot integrand sample
def plot_integrand(times, sample, prefix=''):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'\(t\)')
    ax.plot(times, np.real(sample))
    ax.set_title(r'\(\text{\rm{Re}}\left\{ \langle \sigma^-(t)\sigma^+(0) \rangle e^{-C(t)-\Gamma_\downarrow t/2}\right\}\)')
    plt.savefig('{}/{}integrand{}'.format(FIG_DIR, prefix, save_ext), bbox_inches='tight')
def calculate_width(lambdas, absorption, maximum):
    first_index = np.argmax(absorption > maximum/2)
    second_index = first_index + np.argmax(absorption[first_index:] < maximum/2)
    return np.abs(lambdas[second_index]-lambdas[first_index])
def inspect_spectrum(nus, lambdas, absorption):
    max_absorb_index=np.argmax(absorption)
    l_max_a = lambdas[max_absorb_index]
    f_max_a = nus[max_absorb_index]
    stokes_shift_nm =  lambda0 - l_max_a
    stokes_shift_ev = f_max_a - w0
    max_absorb=absorption[max_absorb_index]
    fwhm_a = calculate_width(lambdas, absorption, max_absorb)
    fwhm_a_ev = calculate_width(nus, absorption, max_absorb)
    print ('Maximum absorption {:.3g} at {:.3g} eV (Stokes shift {:.3g} meV). FWHM {:.3g} meV.'\
            .format(max_absorb, f_max_a, 1000*stokes_shift_ev, 1000*fwhm_a_ev))
    return max_absorb, fwhm_a
def ascending_sort(lambdas, absorption):
    # originally sorted by frequency, so get in nice order for plotting
    # (otherwise matplotlib draws a horizontal line from either end of spectrum)
    to_reorder = [(lambdas[i], absorption[i]) for i in range(len(lambdas))]
    to_reorder.sort() # sorts according to first element by default
    # see https://stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
    lambdas, absorption = list(map(list, zip(*to_reorder))) 
    #print(np.all(np.diff(lambdas)>0))
    return lambdas, absorption
def plot_spectrum(nus, lambdas, absorption, prefix=''):
    fig, ax = plt.subplots()
    label=r'\(a={a}, \nu_c={nuc}, \Gamma_\downarrow={dissipation}, T={TK}\text{{\rm{{K}}}}\)'.format(**args)
    ax.set_title(label, fontsize=16)
    ax.set_xlabel(XLABEL)
    if args['freqs']:
        x_var = XSCALE * (nus-w0)
        ax.plot(x_var, absorption, label=r'\rm{absorption}')
    else:
        x_var, absorption = ascending_sort(lambdas, absorption)
        ax.plot(XSCALE * x_var, absorption)
    ylabel=r'\rm{Intensity (a.u.)}' if not args['no_normalise'] else r'\rm{Intensity}'
    ax.set_ylabel(ylabel)
    if args['emission']:
        deltas = nus-w0
        max_detune=0.05
        i1 = next((i for i,nu in enumerate(nus) if nu>w0-max_detune), None)
        i2 = next((i for i,nu in enumerate(nus) if nu>w0+max_detune), None)
        deltas = deltas[i1:i2]
        crop_absorption = absorption[i1:i2]
        beta = 1/T
        emission = np.exp(- beta * deltas) * crop_absorption
        #if not args['no_normalise']:
        #    emission /= np.max(emission)
        ax.plot(x_var[i1:i2], emission, label=r'\rm{emission}', linewidth=linewidth)
        ax.legend(fontsize=16)
    if PLOT_W0:
        if args['freqs']:
            #ax.axvline(x=XSCALE*w0, c=(1.0,0.0,0.0), alpha=0.65)
            ax.axvline(x=0, c='gray', alpha=.75)
        else:
            ax.axvline(x=XSCALE*lambda0, c='gray', alpha=.75)
            ax.axvline(x=XSCALE*lambda0, c=(1.0,0.0,0.0), alpha=0.65)
        #handles, labels = ax.get_legend_handles_labels()
        ## manually define a new patch 
        ##red_line = Line2D([0], [0], color='green', label='\(\\omega_0={}\\text{{eV}}\)'.format(w0))
        #red_line = Line2D([0], [0], c=(1.0,0.0,0.0), alpha=0.65, 
        #                  label=r'\(\omega_0={}\text{{\rm{{eV}}}}\)'.format(w0), linewidth=linewidth,
        #                  )
        ## handles is a list, so append manual patch
        #handles.append(red_line) 
        ## plot the legend
        #ax.legend(handles=handles, fontsize=16)
    ax.set_xlim(XLIMS)
    plot_fp = f'{FIG_DIR}/{prefix}absorption{save_ext}'
    fig.savefig(plot_fp, bbox_inches='tight')
    print(f'Plotted {plot_fp}')

def multiplot(plot_data, prefix=''):
    fig, ax = plt.subplots()
    ax.set_xlabel(XLABEL)
    ylabel=r'\rm{Absorption (a.u.)}'
    ax.set_ylabel(ylabel)
    max_abs = max([np.max(data[1]) for data in plot_data])
    for data in plot_data:
        to_plot = np.copy(data[1])
        if not args['no_normalise']:
            to_plot /= max_abs
        label='\(a={a}, \\nu_c={nuc}, \\Gamma_\\downarrow={dissipation}\)'.format(**data[-1])
        if args['freqs']:
            ax.plot(XSCALE*(data[0]-w0), to_plot, label=label, linewidth=linewidth)
        else:
            ax.plot(XSCALE*data[0], to_plot, label=label, linewidth=linewidth)
    if PLOT_W0:
        if args['freqs']:
            #ax.axvline(x=XSCALE*w0, c=(1.0,0.0,0.0), alpha=0.65, linewidth=linewidth)
            ax.axvline(x=0, c='gray', alpha=.75)
        else:
            ax.axvline(x=lambda0, c='gray', alpha=.75)
            #ax.axvline(x=lambda0, c=(1.0,0.0,0.0), alpha=0.65, linewidth=linewidth)
        ax.legend(fontsize=14)
        #handles, labels = ax.get_legend_handles_labels()
        #red_line = Line2D([0], [0], c=(1.0,0.0,0.0), alpha=0.65, 
        #                  label='\(\\omega_0={}\\text{{\\rm{{eV}}}}\)'.format(w0),
        #                  linewidth=linewidth)
        #handles.append(red_line) 
        #ax.legend(handles=handles, fontsize=14)
    else:
        ax.legend(fontsize=14)
    ax.set_xlim(XLIMS)
    plot_fp = f'{FIG_DIR}/{prefix}absorption{save_ext}'
    fig.savefig(plot_fp, bbox_inches='tight')
    print(f'Plotted {plot_fp}')

def nm_to_ev(lambdas):
    return (2*pi*hbar*c/e)*1e9*(1/lambdas)    

def save_spectrum(times, correlator_sample, nus, lambdas, absorption):
    stacked_data = np.array([times, correlator_sample, nus, lambdas, absorption, args], dtype='object')
    fp = os.path.join(DATA_DIR, 'T{TK}a{a}nuc{nuc}dt{dt}st{st}d{drive}d{dissipation}.npy'.format(**args))
    np.save(fp, stacked_data)
    print(f'Data saved to {fp} (stacked array)')

def plot_list(fp_list):
    to_plot=[]
    for fp in fp_list:
        stacked_data = np.load(fp, allow_pickle=True)
        times, correlator_sample, nus, lambdas, absorption, loaded_args = stacked_data
        print('Loaded data with a={a}, s={s}, nuc={nuc}, dt={dt}, st={st}, gamma_down={dissipation}.'.format(**loaded_args))
        if args['freqs']:
            to_plot.append([nus, absorption, loaded_args])
        else:
            lambdas, absorption = ascending_sort(lambdas, absorption) 
            to_plot.append([lambdas, absorption, loaded_args])
    #    to_plot.reverse()
    multiplot(to_plot)
def main():
    # Vectorise
    f_vec = np.vectorize(f)
    # Sample times 
    N=int(st/dt+1)
    times=np.linspace(0,st,num=N)
    print('T={:.0f}K: generating {} correlator samples from t={} to t={} (dt={})...'.format(TK, N, 0, st, dt))
    correlator_sample=f_vec(times)
    integrand_sample=correlator_sample*np.exp( -gamma_T * times/2 )
    print('Sampling completed (runtime = {:.0f}s).'.format(time()-t0))
    residual = abs(integrand_sample[-1])
    if residual > 1e-4:
        print('Warning: residual correlator magnitude {:.3g} significant, suggest running for longer times'.format(residual))
    else:
        print('Residual correlator magnitude: {:.3g}'.format(residual))
    print('Performing FFT to get absorption...', end=' ', flush=True)
    if args['correct_fft']:
        nus, fft = fft_corr(times, integrand_sample, over_sample_factor=args['over_sample_factor'])
        absorption = (gn/2)**2*2*np.real(dt * fft)
    else:
        min_fft_length=args['over_sample_factor']*len(times) # i.e. corresponds to st=1000 # pad with zeros after actual st=times[-1]
        fft_length=max(min_fft_length, len(times))
        nus= 2*pi*np.fft.fftshift(np.fft.fftfreq(fft_length, d=dt))+w0
        # No LM coupling, convention of +iwt in exponent - this is correct! See Mollow 1972 Eq. 4.10  (normalisation to 2Pi)
        absorption=(gn/2)**2*2*np.real(dt*np.fft.fftshift(np.fft.ifft(integrand_sample, norm='forward', n=fft_length)))
        # other convention
        #absorption=2*np.real(dt*np.fft.fftshift(np.fft.fft(integrand_sample, n=fft_length)))
    nus_hz=nus*EV_TO_HZ
    with np.errstate(divide='ignore'):
        #lambdas=1e9*(2*pi*c/nus_hz) # wavelengths in nm
        lambdas=HZ_TO_NM*(1/nus_hz)
    print('done (runtime = {:.0f}s).'.format(time()-t0))
    max_absorb, width = inspect_spectrum(nus, lambdas, absorption)
    if not args['no_normalise']:
        absorption=absorption/max_absorb
    plot_integrand(times, integrand_sample, prefix='')
    plot_spectrum(nus, lambdas, absorption, prefix='')
    save_spectrum(times, correlator_sample, nus, lambdas, absorption)

if args['plot'] is not None:
    plot_list(args['plot'])
else:
    main()
