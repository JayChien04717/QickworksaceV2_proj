import numpy as np
from scipy.optimize import least_squares


def complex_fit(f, xdata, ydata, p0=None, weights=None, **kwargs):
    """Wrapper around scipy least_square for complex functions.

    Parameters
    ----------
    f : Any
        Value for ``f``.
    xdata : Any
        Independent-variable data.
    ydata : Any
        Dependent-variable data.
    p0 : Any, default: None
        Value for ``p0``.
    weights : Any, default: None
        Value for ``weights``.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        Result of the operation.

    Raises
    ------
    ValueError
        If the operation cannot be completed.
    """

    if (np.array(ydata).size - len(p0)) <= 0:
        raise ValueError(
            "yData length should be greater than the number of parameters."
        )

    def residuals(params, x, y):
        """Computes the residual for the least square algorithm

        Parameters
        ----------
        params : Any
            Value for ``params``.
        x : Any
            Independent-variable values.
        y : Any
            Dependent-variable values.

        Returns
        -------
        Any
            Result of the operation.
        """
        if weights is not None:
            diff = weights * f(x, *params) - y
        else:
            diff = f(x, *params) - y
        flat_diff = np.zeros(diff.size * 2, dtype=np.float64)
        flat_diff[0 : flat_diff.size : 2] = diff.real
        flat_diff[1 : flat_diff.size : 2] = diff.imag
        return flat_diff

    kwargs_ls = kwargs.copy()
    kwargs_ls.setdefault("max_nfev", 1000)
    kwargs_ls.setdefault("ftol", 1e-2)
    opt_res = least_squares(residuals, p0, args=(xdata, ydata), **kwargs_ls)

    jac = opt_res.jac
    cost = opt_res.cost

    pcov = np.linalg.inv(jac.T.dot(jac))
    pcov *= cost / (np.array(ydata).size - len(p0))

    popt = opt_res.x

    return popt, pcov


def guess_edelay_from_gradient(freq, signal, n=-1):

    """Return the guess edelay from gradient result.

    Parameters
    ----------
    freq : Any
        Value for ``freq``.
    signal : Any
        Value for ``signal``.
    n : Any, default: -1
        Value for ``n``.

    Returns
    -------
    Any
        Result of the operation.
    """
    dtheta = np.mean(np.angle(signal[-n:] / zeros2eps(signal[:n])))
    df = np.mean(np.diff(freq))

    return dtheta / df / 2 / np.pi


def smooth_gradient(signal):
    """Return the smooth gradient result.

    Parameters
    ----------
    signal : Any
        Value for ``signal``.

    Returns
    -------
    Any
        Result of the operation.
    """
    def dnormaldx(x, x_0, sigma):
        """Return the dnormaldx result.

        Parameters
        ----------
        x : Any
            Independent-variable values.
        x_0 : Any
            Value for ``x_0``.
        sigma : Any
            Value for ``sigma``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return -(x - x_0) * np.exp(-0.5 * ((x - x_0) / sigma) ** 2)

    conv_kernel_size = max(min(100, signal.size // 20), 2)

    conv_kernel = dnormaldx(
        x=np.arange(0.5, conv_kernel_size + 0.5, 1),
        x_0=conv_kernel_size / 2,
        sigma=conv_kernel_size / 8,
    )

    gradient = np.convolve(signal, conv_kernel, "same")
    gradient[: conv_kernel_size // 2] = gradient[
        conv_kernel_size // 2 : 2 * (conv_kernel_size // 2)
    ][::-1]
    gradient[-(conv_kernel_size // 2) :] = gradient[
        -2 * (conv_kernel_size // 2) : -(conv_kernel_size // 2)
    ][::-1]

    return gradient


eps = np.finfo(float).eps


def zeros2eps(x):

    """args:
                x: float, complex, or numpy array

            return:
                y: numpy array

            replace the zeros of a float or numpy array bien the smallest float number

    Parameters
    ----------
    x : Any
        Independent-variable values.

    Returns
    -------
    Any
        Result of the operation.
    """

    y = np.array(x)
    y[np.abs(y) < eps] = eps

    return y


def dB(x):

    """Return the dB result.

    Parameters
    ----------
    x : Any
        Independent-variable values.

    Returns
    -------
    Any
        Result of the operation.
    """
    return 20 * np.log10(np.abs(x))


def deg(x):

    """Return the deg result.

    Parameters
    ----------
    x : Any
        Independent-variable values.

    Returns
    -------
    Any
        Result of the operation.
    """
    return np.angle(x) * 180 / np.pi


def get_prefix(x):

    """Return prefix.

    Parameters
    ----------
    x : Any
        Independent-variable values.

    Returns
    -------
    Any
        Result of the operation.
    """
    prefix = [
        "y",  # yocto
        "z",  # zepto
        "a",  # atto
        "f",  # femto
        "p",  # pico
        "n",  # nano
        "u",  # micro
        "m",  # mili
        "",
        "k",  # kilo
        "M",  # mega
        "G",  # giga
        "T",  # tera
        "P",  # peta
        "E",  # exa
        "Z",  # zetta
        "Y",  # yotta
    ]

    max_x = np.abs(np.max(x))

    if max_x > 10 * eps:

        index = int(np.log10(max_x) / 3 + 8)
        return (x * 10 ** (-3 * (index - 8)), prefix[index])

    else:

        return (0, "")


def get_prefix_str(x, precision=2):

    """Return prefix str.

    Parameters
    ----------
    x : Any
        Independent-variable values.
    precision : Any, default: 2
        Value for ``precision``.

    Returns
    -------
    Any
        Result of the operation.
    """
    return "%.{}f %s".format(precision) % get_prefix(x)

