import qutip as qt
import numpy as np


def flux_to_phi(x, flux_range, phi_range):
    """Map an external flux value to a reduced flux bias phi in [0, 0.5].

    Parameters
    ----------
    x : Any
        Independent-variable values.
    flux_range : Any
        Value for ``flux_range``.
    phi_range : Any
        Value for ``phi_range``.

    Returns
    -------
    Any
        Result of the operation.
    """
    src_min, src_max = flux_range
    tgt_min, tgt_max = phi_range
    val = tgt_min + (x - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    val = val % 1
    return val if val <= 0.5 else 1 - val


def phi_to_flux(phi, flux_range, phi_range):
    """Map a reduced flux bias phi back to an external flux value.

    Parameters
    ----------
    phi : Any
        Value for ``phi``.
    flux_range : Any
        Value for ``flux_range``.
    phi_range : Any
        Value for ``phi_range``.

    Returns
    -------
    Any
        Result of the operation.
    """
    src_min, src_max = phi_range
    tgt_min, tgt_max = flux_range
    flux_val = tgt_min + (phi - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    return flux_val


class Fluxonium:
    """
    Fluxonium qubit Hamiltonian and transition frequency calculator.

    Parameters
    ----------
    EJ : float
        Josephson energy in GHz.
    EC : float
        Charging energy in GHz.
    EL : float
        Inductive energy in GHz.
    dimention : int
        Hilbert space dimension (number of Fock states).
    flux : float
        Reduced external flux bias phi in [0, 0.5].
    """

    def __init__(self, EJ, EC, EL, dimention, flux) -> None:
        """Initialize the Fluxonium instance.

        Parameters
        ----------
        EJ : Any
            Value for ``EJ``.
        EC : Any
            Value for ``EC``.
        EL : Any
            Value for ``EL``.
        dimention : Any
            Value for ``dimention``.
        flux : Any
            Value for ``flux``.
        """
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.dim = dimention
        self.phi = flux

    @property
    def phi_osc(self):
        """Return the phi osc result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return ((8 * self.EC) / self.EL) ** (1 / 4)

    @property
    def creation(self):
        """Return the creation result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return qt.create(self.dim)

    @property
    def destroy(self):
        """Return the destroy result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return qt.destroy(self.dim)

    @property
    def n_op(self):
        """Return the n op result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return (-1j / (np.sqrt(2) * self.phi_osc)) * (self.creation - self.destroy)

    @property
    def phi_op(self):
        """Return the phi op result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return (self.phi_osc / np.sqrt(2)) * (self.creation + self.destroy)

    def J_term(self):
        """Return the J term result.

        Returns
        -------
        Any
            Result of the operation.
        """
        phi_ext_op = qt.qeye(self.dim) * (2 * np.pi * self.phi)
        return -self.EJ * (self.phi_op - phi_ext_op).cosm()

    def C_term(self):
        """Return the C term result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return 4 * self.EC * self.n_op**2

    def L_term(self):
        """Return the L term result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return 0.5 * self.EL * self.phi_op**2

    def hamiltonian(self):
        """Return the hamiltonian result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self.J_term() + self.C_term() + self.L_term()

    def f01(self):
        """Return the f01 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        energies = self.hamiltonian().eigenenergies()
        return energies[1] - energies[0]

    def f12(self):
        """Return the f12 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        energies = self.hamiltonian().eigenenergies()
        return energies[2] - energies[1]

    def f02(self):
        """Return the f02 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        energies = self.hamiltonian().eigenenergies()
        return energies[2] - energies[0]

    def f03(self):
        """Return the f03 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        energies = self.hamiltonian().eigenenergies()
        return energies[3] - energies[0]

    def fij(self, i, j):
        """Return the fij result.

        Parameters
        ----------
        i : Any
            Value for ``i``.
        j : Any
            Value for ``j``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        if i < j:
            raise ValueError(f"Invalid transition: i={i} must be >= j={j}")
        energies = self.hamiltonian().eigenenergies()
        return energies[i] - energies[j]

    def cooling_to_g(self, fr):
        """Return the cooling to g result.

        Parameters
        ----------
        fr : Any
            Value for ``fr``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return {"f12": self.fij(2, 1), "f0g1": fr - self.fij(2, 0)}

    def cooling_to_e(self, fr):
        """Return the cooling to e result.

        Parameters
        ----------
        fr : Any
            Value for ``fr``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return {"f03": self.fij(3, 0), "fhe1": fr - self.fij(3, 1)}
