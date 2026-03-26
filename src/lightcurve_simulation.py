import numpy as np
import astropy.constants as consts
import astropy.units as u
from stingray.simulator import Simulator
from binlite.flux import periodic_flux_series_from_bad
from binlite import AccretionSeries, BinaryAlphaDisk

# ================= PHYSICAL CONSTANTS =================
c = consts.c.cgs
h = consts.h.cgs.value

# ================= CORE PHYSICS =================
def compute_photon_flux(BH_mass, fedd, dl_pc, obs_nu):
    """
    Compute photon flux (photons/s/cm^2)
    """
    L_edd = 1.26e38 * BH_mass
    L = fedd * L_edd
    dl_cm = dl_pc * 3.086e18
    flux_mean = L / (4 * np.pi * dl_cm**2)
    E_photon = h * obs_nu
    return flux_mean / E_photon


def simulate_red_noise(mean_rate, N, rms, dt, seed=None):
    """
    Generate red noise light curve
    """
    sim = Simulator(
        N=N,
        mean=mean_rate,
        rms=rms,
        dt=dt,
        random_state=seed,
        poisson=False
    )
    return sim.simulate(2).counts


# ================= BINNING =================
def bin_to_fixed_length(arr, target_len):
    """
    Photon-conserving binning
    """
    bin_width = len(arr) // target_len
    return arr.reshape(target_len, bin_width).sum(axis=1)


# ================= SINGLE AGN =================
def generate_single_agn_curve(
    BH_mass,
    fedd,
    dl_pc,
    obs_nu,
    ZTF_area,
    N,
    dt,
    target_len,
    seed=None
):
    photon_flux = compute_photon_flux(BH_mass, fedd, dl_pc, obs_nu)
    photon_rate = photon_flux * ZTF_area

    noise = simulate_red_noise(
        mean_rate=1.0,
        N=N,
        rms=0.2,
        dt=dt,
        seed=seed
    )

    rate = np.clip(photon_rate * noise, 0, None)
    lambda_native = rate * dt

    binned = bin_to_fixed_length(lambda_native, target_len)
    counts = np.random.poisson(binned)

    return counts


# ================= BINARY AGN =================
def generate_binary_agn_curve(
    ecc,
    n_orbits,
    BH_mass,
    period_yr,
    fedd,
    dl_pc,
    obs_nu,
    ZTF_area,
    N,
    dt,
    target_len,
    seed=None
):
    photon_flux = compute_photon_flux(BH_mass, fedd, dl_pc, obs_nu)
    photon_rate = photon_flux * ZTF_area

    # ----- Noise -----
    noise = simulate_red_noise(1.0, N, 0.2, dt, seed)
    noise_rate = np.clip(photon_rate * noise, 0, None)
    noise_lambda = bin_to_fixed_length(noise_rate * dt, target_len)

    # ----- Signal -----
    accretion = AccretionSeries(ecc, n_modes=29, n_orbits=n_orbits)
    disk = BinaryAlphaDisk(
        ecc,
        period_yr,
        BH_mass,
        dl_pc,
        eddington_ratio=fedd,
        retrograde=accretion.is_retro
    )

    raw_flux = periodic_flux_series_from_bad(obs_nu, accretion, disk)

    signal_rate = raw_flux / (h * obs_nu) * ZTF_area
    signal_rate = np.clip(signal_rate, 0, None)

    # interpolate to match time grid
    time_native = np.linspace(0, N * dt, N, endpoint=False)
    time_model = np.linspace(0, N * dt, len(signal_rate), endpoint=False)

    signal_interp = np.interp(time_native, time_model, signal_rate)
    signal_lambda = bin_to_fixed_length(signal_interp * dt, target_len)

    # ----- Combine -----
    total_lambda = noise_lambda + signal_lambda
    counts = np.random.poisson(total_lambda)

    return counts# Lightcurve simulation module
