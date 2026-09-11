import numpy as np
from copy import deepcopy
from .plot import plot

if __name__ == "__main__":

    from utils import (
        zeros2eps,
        get_prefix_str,
    )

else:

    from .utils import (
        zeros2eps,
        get_prefix_str,
    )


def transmission(freq, f_0, kappa):

    """Return the transmission result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    f_0 : Any
        Value for ``f_0``.
    kappa : Any
        Value for ``kappa``.

    Returns
    -------
    Any
        Result of the operation.
    """
    num = 1
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection(freq, f_0, kappa, kappa_c_real, phi_0=0):

    """Return the reflection result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    f_0 : Any
        Value for ``f_0``.
    kappa : Any
        Value for ``kappa``.
    kappa_c_real : Any
        Value for ``kappa_c_real``.
    phi_0 : Any, default: 0
        Value for ``phi_0``.

    Returns
    -------
    Any
        Result of the operation.
    """
    num = 2j * (freq - f_0) + kappa - 2*kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    """Return the reflection mismatched result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    f_0 : Any
        Value for ``f_0``.
    kappa : Any
        Value for ``kappa``.
    kappa_c_real : Any
        Value for ``kappa_c_real``.
    phi_0 : Any
        Value for ``phi_0``.

    Returns
    -------
    Any
        Result of the operation.
    """
    return reflection(freq, f_0, kappa, kappa_c_real, phi_0)


def hanger(freq, f_0, kappa, kappa_c_real, phi_0=0):

    """Return the hanger result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    f_0 : Any
        Value for ``f_0``.
    kappa : Any
        Value for ``kappa``.
    kappa_c_real : Any
        Value for ``kappa_c_real``.
    phi_0 : Any, default: 0
        Value for ``phi_0``.

    Returns
    -------
    Any
        Result of the operation.
    """
    num = 2j * (freq - f_0) + kappa - kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def hanger_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    """Return the hanger mismatched result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    f_0 : Any
        Value for ``f_0``.
    kappa : Any
        Value for ``kappa``.
    kappa_c_real : Any
        Value for ``kappa_c_real``.
    phi_0 : Any
        Value for ``phi_0``.

    Returns
    -------
    Any
        Result of the operation.
    """
    return hanger(freq, f_0, kappa, kappa_c_real, phi_0)

resonator_dict = {
    "transmission": transmission,
    "t": transmission,
    "reflection": reflection,
    "r": reflection,
    "reflection_mismatched": reflection_mismatched,
    "rm": reflection_mismatched,
    "hanger": hanger,
    "h": hanger,
    "hanger_mismatched": hanger_mismatched,
    "hm": hanger_mismatched,
}

def get_fit_function(geometry, amplitude=True, edelay=True):
    """Return fit function.

    Parameters
    ----------
    geometry : Any
        Value for ``geometry``.
    amplitude : Any, default: True
        Value for ``amplitude``.
    edelay : Any, default: True
        Value for ``edelay``.

    Returns
    -------
    Any
        Result of the operation.

    Raises
    ------
    Exception
        If the operation cannot be completed.
    """
    if isinstance(geometry, str):
        resonator_func = resonator_dict[geometry]
    else:
        resonator_func = geometry

    if not amplitude and not edelay:
        return resonator_func

    elif amplitude and not edelay:
        def fit_func(*args):
            """Fit func.

            Parameters
            ----------
            *args : Any
                Additional positional arguments.

            Returns
            -------
            Any
                Result of the operation.
            """
            return resonator_func(*args[:-2]) * (args[-2] + 1j * args[-1])

        return fit_func

    elif not amplitude and edelay:
        def fit_func(*args):
            """Fit func.

            Parameters
            ----------
            *args : Any
                Additional positional arguments.

            Returns
            -------
            Any
                Result of the operation.
            """
            return resonator_func(*args[:-1]) * np.exp(2j * np.pi * args[-1] * args[0])

        return fit_func

    elif amplitude and edelay:
        def fit_func(*args):
            """Fit func.

            Parameters
            ----------
            *args : Any
                Additional positional arguments.

            Returns
            -------
            Any
                Result of the operation.
            """
            return (
                resonator_func(*args[:-3])
                * (args[-3] + 1j * args[-2])
                * np.exp(2j * np.pi * args[-1] * args[0])
            )

        return fit_func

    else:

        raise Exception("Unreachable")

class ResonatorParams(object):
    def __init__(self, params, geometry, freq = None, signal = None):

        """Initialize the ResonatorParams instance.

        Parameters
        ----------
        params : Any
            Value for ``params``.
        geometry : Any
            Value for ``geometry``.
        freq : Any, default: None
            Value for ``freq``.
        signal : Any, default: None
            Value for ``signal``.
        """
        self.resonator_func = resonator_dict[geometry]
        self.params = params

        self.freq = freq
        self.signal = signal

        if self.resonator_func == transmission:
            self.f_0_index = 0
            self.kappa_index = 1
            if len(self.params) in [4, 5]:
                self.re_a_in_index = 2
                self.im_a_in_index = 3
            if len(self.params) in [3, 5]:
                self.edelay_index = -1

        if self.resonator_func in [reflection, hanger]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            if len(self.params) in [5, 6]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [4, 6]:
                self.edelay_index = -1

        if self.resonator_func in [reflection_mismatched, hanger_mismatched]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            self.phi_0_index = 3
            if len(self.params) in [6, 7]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [5, 7]:
                self.edelay_index = -1

    def tolist(self):
        """Return the tolist result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return np.array(self.params)

    @property
    def f_0(self):
        """Return the f 0 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "f_0_index"):
            return self.params[self.f_0_index]
        else:
            return None

    @property
    def kappa(self):
        """Return the kappa result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "kappa_index"):
            return self.params[self.kappa_index]
        else:
            None

    @property
    def kappa_i(self):
        """Return the kappa i result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "kappa_index") and hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_index] - self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c_real(self):
        """Return the kappa c real result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c(self):
        """Return the kappa c result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self.kappa_c_real

    @property
    def a_in(self):
        """Return the a in result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "re_a_in_index") and hasattr(self, "im_a_in_index"):
            return (
                self.params[self.re_a_in_index] + 1j * self.params[self.im_a_in_index]
            )
        else:
            return None

    @property
    def re_a_in(self):
        """Return the re a in result.

        Returns
        -------
        Any
            Result of the operation.
        """
        a_in = self.a_in
        if a_in is not None:
            return np.real(a_in)
        else:
            return None

    @property
    def im_a_in(self):
        """Return the im a in result.

        Returns
        -------
        Any
            Result of the operation.
        """
        a_in = self.a_in
        if a_in is not None:
            return np.imag(a_in)
        else:
            return None

    @property
    def edelay(self):
        """Return the edelay result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "edelay_index"):
            return self.params[self.edelay_index]
        else:
            return None

    @property
    def phi_0(self):
        """Return the phi 0 result.

        Returns
        -------
        Any
            Result of the operation.
        """
        if hasattr(self, "phi_0_index"):
            return self.params[self.phi_0_index]
        else:
            return None

    def str(self, latex=False, separator=", ", precision=2, only_f_and_kappa=False, f_precision=2, red_warning = False):
        """Return the str result.

        Parameters
        ----------
        latex : Any, default: False
            Value for ``latex``.
        separator : Any, default: ', '
            Value for ``separator``.
        precision : Any, default: 2
            Value for ``precision``.
        only_f_and_kappa : Any, default: False
            Value for ``only_f_and_kappa``.
        f_precision : Any, default: 2
            Value for ``f_precision``.
        red_warning : Any, default: False
            Value for ``red_warning``.

        Returns
        -------
        Any
            Result of the operation.
        """
        kappa = {False: "kappa/2pi", True: r"$\kappa/2\pi$"}
        kappa_c = {False: "kappa_c/2pi", True: r"$\kappa_c/2\pi$"}
        f_0 = {False: "f_0", True: r"$f_0$"}
        phi_0 = {False: "phi_0", True: r"$\varphi_0$"}

        if self.edelay is not None:
            edelay_str = "%sedelay = %ss" % (separator, get_prefix_str(self.edelay, precision))
        else:
            edelay_str = ""

        if self.resonator_func == transmission:
            kappa_str = r"%s%s = %sHz" % (
                separator,
                kappa[latex],
                get_prefix_str(self.kappa, precision),
            )
        else:
            kappa_str = r"%s%s = %sHz%s%s = %sHz" % (
                separator,
                kappa[latex],
                get_prefix_str(self.kappa, precision),
                separator,
                kappa_c[latex],
                get_prefix_str(self.kappa_c, precision),
            )

        if self.resonator_func in [hanger_mismatched, reflection_mismatched]:
            if red_warning and self.phi_0 is not None and np.abs(self.phi_0) > 0.25:
                phi_0_str = r"%s = %0.2f rad" % (phi_0[latex], self.phi_0)
                phi_0_str = "%s/!\\ "%separator + phi_0_str + " /!\\"
            else:
                phi_0_str = r"%s%s = %0.2f rad" % (separator, phi_0[latex], self.phi_0)
        else:
            phi_0_str = ""

        f_0_str = r"%s = %sHz" % (f_0[latex], get_prefix_str(self.f_0, f_precision))

        if only_f_and_kappa:
            return f_0_str + kappa_str
        else:
            return f_0_str + kappa_str + phi_0_str + edelay_str

    def __str__(self) -> str:
        """Return a human-readable representation.

        Returns
        -------
        str
            Result of the operation.
        """
        return self.str()

    def __repr__(self):
        """Return a human-readable representation.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self.str()

    def __call__(self, freq, *args, **kwargs):
        """Return the call result.

        Parameters
        ----------
        freq : Any
            Value for ``freq``.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Result of the operation.
        """
        amplitude = self.a_in is not None
        edelay = self.edelay is not None

        fit_func = get_fit_function(self.resonator_func, amplitude, edelay)

        if len(args) == 0 and len(kwargs) == 0:
            params = self.params
        else:
            if len(kwargs) == 0:
                params = np.copy(self.params)
                params[:len(args)] = args
            else:
                resonator = deepcopy(self)
                for key in kwargs:
                    resonator.params[resonator.__dict__[key + "_index"]] = kwargs[key]
                resonator.params[:len(args)] = args
                params = resonator.params
        
        return fit_func(freq, *params)
    
    def plot(
            self,
            fig = None,
            plot_not_corrected = True,
            font_size = None,
            plot_circle = True,
            center_freq = False,
            only_f_and_kappa = False,
            precision = 2,
            alpha_fit = 1.0,
            style = 'Normal',
            title = None,
            params = None,
        ):
        """Plot the operation.

        Parameters
        ----------
        fig : Any, default: None
            Matplotlib figure to update.
        plot_not_corrected : Any, default: True
            Value for ``plot_not_corrected``.
        font_size : Any, default: None
            Value for ``font_size``.
        plot_circle : Any, default: True
            Value for ``plot_circle``.
        center_freq : Any, default: False
            Value for ``center_freq``.
        only_f_and_kappa : Any, default: False
            Value for ``only_f_and_kappa``.
        precision : Any, default: 2
            Value for ``precision``.
        alpha_fit : Any, default: 1.0
            Value for ``alpha_fit``.
        style : Any, default: 'Normal'
            Value for ``style``.
        title : Any, default: None
            Value for ``title``.
        params : Any, default: None
            Value for ``params``.
        """
        plot(
            self.freq,
            self.signal,
            self(self.freq),
            fig=fig,
            fit_params=self,
            params=params,
            plot_not_corrected=plot_not_corrected,
            font_size=font_size,
            plot_circle=plot_circle,
            center_freq=center_freq,
            only_f_and_kappa=only_f_and_kappa,
            precision=precision,
            alpha_fit=alpha_fit,
            style=style,
            title=title
        )


if __name__ == "__main__":

    f_0 = 3.8e9
    kappa_i = 50e6
    kappa_c = 150e6
    a_in = 1 + 1j
    edelay = 32e-9
    phi_0 = 4
    geometry = "hm"

    params = [f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in), edelay]

    params = ResonatorParams(params, geometry)

    print(params)

    f_0 = 3.8e9
    kappa_i = 50e6
    kappa_c = 150e6
    a_in = 1 + 1j
    edelay = 32e-9
    geometry = "r"

    params = [f_0, kappa_i, kappa_c, np.real(a_in), np.imag(a_in), edelay]

    params = ResonatorParams(params, geometry)

    print(params)

    f_0 = 3.8e9
    kappa = 50e6
    a_in = 1 + 1j
    geometry = "t"

    params = [f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)]

    params = ResonatorParams(params, geometry)
