import qutip
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# reduced Planck constant
hbar = 1  # or 1.054571817 * 10 ** -34 J * s

# bare electron mass
m0 = 9.109 * 10 ** -39  # kg

# Spin-3/2 matrices
Sx = 0.5 * hbar * np.matrix([
    [0, np.sqrt(3), 0, 0],
    [np.sqrt(3), 0, 2, 0],
    [0, 2, 0, np.sqrt(3)],
    [0, 0, np.sqrt(3), 0]
])

Sy = 0.5 * hbar * np.matrix([
    [0, complex(0, np.sqrt(3)), 0, 0],
    [complex(0, np.sqrt(3)), 0, -2j, 0],
    [0, 2j, 0, complex(0, -np.sqrt(3))],
    [0, 0, complex(0, np.sqrt(3)), 0]
])

Sz = 0.5 * hbar * np.matrix([
    [3, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -3]
])

# Spin-1/2 matrices
Sx_half = 0.5 * hbar * np.matrix([
    [0, 1],
    [1, 0]
])

Sz_half = 0.5 * hbar * np.matrix([
    [1, 0],
    [0, -1]
])

Sy_half = complex(0, -1) * Sx_half * Sz_half

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
def Hlutt(gamma1, gamma2, gamma3, K):
    return (gamma1 + 5 * gamma2 / 2) * ((hbar ** 2 * np.dot(K, K)) / (2 * m0)) \
    + gamma2 * (hbar ** 2 / m0) * (kx ** 2 * Sx ** 2 + ky ** 2 * Sy ** 2 + kz ** 2 * Sz ** 2) \
        + gamma3 * (hbar ** 2 / (2 * m0)) * (((kx * ky + ky * kx) / 2) * ((Sx * Sy + Sy * Sx) / 2)) \
            * (((ky * kz + kz * ky) / 2) * ((Sy * Sz + Sz * Sy) / 2)) * (((kz * kx + kx * kz) / 2) * ((Sz * Sx + Sx * Sz) / 2))


# elmentary charge
e = 1.602176634 * 10 ** -19  # C

# Bohr magneton
mB = e * hbar / (2 * m0)  # J / T

# vacuum permeability constant
m0 = 1.256637062 * 10 ** -6  # N * A ** -2

# electron gyromagnetic ratio
gamma_s = -2 * mB  # J / T

# nuclear gyromagnetic ratio
gamma_l = 1  # C / kg

# nuclear core charge
Z = 1  # C

# fine structure constant
alpha = 1 / 137

# speed of light in vacuum
c = 299792458  # m / s

# Bohr radius
aB = hbar / (m0 * c * alpha)  # m

# Thompson radius
rT = Z * alpha ** 2 * aB  # m


# Localized functions
def fT(r):
    return r / (r + rT / 2)


def d_dr_fT(r):
    return (rT / 2) / (rT / 2 + r) ** 2


prefix = (m0 / 4 * np.pi) * np.abs(gamma_s) * gamma_l


# Fermi contact interaction Hamiltonian
def Hc(R, Rl, S, Il):
    """
    Returns the Fermi contact interaction between an electron at position R with spin S and a nucleus with position Rl and spin Il.
    """
    return prefix * (8 * np.pi / 3) * ((1 / (4 * np.pi * R ** 2)) * d_dr_fT(R - Rl) * np.dot(S, Il))


# Dipolar tensor
def D(R):
    matrix = []
    for i in range(len(R)):
        row = []
        for j in range(len(R)):
            term = ((3 * R[i] * R[j] - np.dot(R, R) * (i == j)) / np.dot(R, R) ** (5/2)) * fT(R)
            row.append(term)
        matrix.append(row)
    
    return np.array(matrix)


# Dipolar coupling Hamiltonian
def Hdip(R, Rl, S, Il):
    """
    Returns the diploar coupling between an electron at position R with spin S and a nucleus with position Rl and spin Il.
    """ 
    return prefix * np.dot(S, np.dot(D(R - Rl), Il))  # TODO: how to make dimensions match dipolar tensor?


# Orbital coupling Hamiltonian
def Horb(R, Rl, Ll, Il):
    """
    Returns the orbital coupling between an electron at position R with orbital angular momentum Ll about site l and a nucleus with position Rl and spin Il.
    """ 
    return prefix * (1 / np.abs(R - Rl) ** 3) * fT(np.abs(R - Rl)) * np.dot(Ll, Il)


# Dirac equation
def Hhf(R, S, nuclei):
    """
    Nuclei a list of [Rl, Ll, Il].
    """
    res = Hc(R, nuclei[0][0], S, nuclei[0][2]) + Hdip(R, nuclei[0][0], S, nuclei[0][2]) + Horb(R, nuclei[0][0], nuclei[0][1], nuclei[0][2])
    i = 1
    while i <= len(nuclei):
        res += Hc(R, nuclei[i][0], S, nuclei[i][2]) + Hdip(R, nuclei[i][0], S, nuclei[i][2]) + Horb(R, nuclei[i][0], nuclei[i][1], nuclei[i][2])
        i += 1

    return res


# single confined particle Zeeman interaction
def Hzeeman(B):
    S = [Sx_half, Sy_half, Sz_half]
    res = 0
    for dimension in range(len(B)):
        res += gamma_s * mB * B[dimension] * S[dimension]  # TODO: do we use gamma_s for holes?


nuclei = [[np.array([0 * 10 ** -6, 0 * 10 ** -6, 0 * 10 ** -6]), np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])]]


def fermi_contact_interaction(dimension):
    data = []
    for i in range(10):
        row = []
        for j in range(10):
            electron = [np.array([i * 10 ** -6, j * 10 ** -6, 0 * 10 ** -6]), np.array([1, 0, 0, 0])]
            ham = Hc(electron[0], nuclei[0][0], electron[1], nuclei[0][2])
            row.append(ham[dimension])
        data.append(row)

    return data


def orbital_coupling(dimension):
    data = []
    for i in range(10):
        row = []
        for j in range(10):
            electron = [np.array([i * 10 ** -6, j * 10 ** -6, 0 * 10 ** -6]), np.array([1, 0, 0, 0])]
            ham = Horb(electron[0], nuclei[0][0], nuclei[0][1], nuclei[0][2])
            row.append(ham[dimension])
        data.append(row)

    return data


def dipolar_coupling(dimension):
    data = []
    for i in range(10):
        row = []
        for j in range(10):
            electron = [np.array([i * 10 ** -6, j * 10 ** -6, 0 * 10 ** -6]), np.array([1, 0, 0, 0])]
            ham = Hdip(electron[0], nuclei[0][0], electron[1], nuclei[0][2])
            row.append(ham[dimension])
        data.append(row)

    return data


def plot_interaction(interaction):
    colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    my_cmap = ListedColormap(colors, name="my_cmap")

    def plot_examples(colormap, datasets):
        """
        Helper function to plot data with associated colormap.
        """
        n = len(datasets)
        fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                                constrained_layout=True, squeeze=False)
        for [ax, data] in zip(axs.flat, datasets):
            psm = ax.pcolormesh(data, cmap=colormap, rasterized=True, vmin=0, vmax=1 * 10 ** 30)
            fig.colorbar(psm, ax=ax)
        plt.show()

    plot_examples(my_cmap, [interaction(0), interaction(1)])


if __name__ == "__main__":
    plot_interaction(fermi_contact_interaction)
    # plot_interaction(orbital_coupling)
    # plot_interaction(dipolar_coupling)
    print("done")
