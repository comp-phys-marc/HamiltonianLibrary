import qutip
import numpy as np

# reduced Planck constant
hbar = 1 # or 1.054571817 * 10 ** -34 J * s

# bare electron mass
m0 = 9.109 * 10 ** -39 # kg

# Spin-3/2 matrices
Sx = 0.5 * hbar * np.array([
    [0, np.sqrt(3), 0, 0],
    [np.sqrt(3), 0, 2, 0],
    [0, 2, 0, np.sqrt(3)],
    [0, 0, np.sqrt(3), 0]
])

Sy = 0.5 * hbar * np.array([
    [0, complex(0, np.sqrt(3)), 0, 0],
    [complex(0, np.sqrt(3)), 0, -2j, 0],
    [0, 2j, 0, complex(0, -np.sqrt(3))],
    [0, 0, complex(0, np.sqrt(3)), 0]
])

Sz = 0.5 * hbar * np.array([
    [3, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -3]
])

# Luttinger parameters
gamma1 = 1
gamma2 = 1
gamma3 = 1

# crystal momentum
kx = 1 # kg * m / s
ky = 1 # kg * m / s
kz = 1 # kg * m / s

K = np.array([kx, ky, kz])

# Luttinger Hamiltonian for hole spins
H = (gamma1 + 5 * gamma2 / 2) * ((hbar ** 2 * K ** 2) / (2 * m0)) \
    + gamma2 * (hbar ** 2 / m0) * (kx ** 2 * Sx ** 2 + ky ** 2 * Sy ** 2 + kz ** 2 * Sz ** 2) \
        + gamma3 * (hbar ** 2 / (2 * m0)) * (((kx * ky + ky * kx) / 2) * ((Sx * Sy + Sy * Sx) / 2)) \
            * (((ky * kz + kz * ky) / 2) * ((Sy * Sz + Sz * Sy) / 2)) * (((kz * kx + kx * kz) / 2) * ((Sz * Sx + Sx * Sz) / 2))



# elmentary charge
e = 1.602176634 * 10 ** -19 # C

# Bohr magneton
mB = e * hbar / (2 * m0) # J / T

# vacuum permeability constant
m0 = 1.256637062 * 10 ** -6 # N * A ** -2

# electron gyromagnetic ratio
gamma_s = -2 * mB # J / T

# nuclear gyromagnetic ratio
gamma_l = 1 # C / kg

# nuclear core charge
Z = 1 # C

# fine structure constant
alpha = 1 / 137

# speed of light in vacuum
c = 299792458 # m / s

# Bohr radius
aB = hbar / (m0 * c * alpha) # m

# Thompson radius
rT = Z * alpha ** 2 * aB # m

# Localized functions
def fT(r):
    return r / (r + rT / 2)

def d_dr_fT(r):
    return (rT / 2) / (rT / 2 + r) ** 2

# Fermi contact interaction Hamiltonian
def Hc(R, Rl, S, Il):
    """
    Returns the Fermi contact interaction between an electron at position R with spin S and a nucleus with position Rl and spin Il.
    """ 
    return (m0 / 4 * np.pi) * (8 * np.pi / 3) * np.abs(gamma_s) * gamma_l * ((1 / (4 * np.pi * r ** 2)) * d_dr_fT(R - Rl) * S * Il)
