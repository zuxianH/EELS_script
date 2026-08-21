import numpy as np


def tacaw_theta_grid(V, Lx, Ly, theta_max=100):
    """
    Generate TACAW scattering angle grid.

    Parameters
    ----------
    V : float
        Electron accelerating voltage in volts.

    Lx, Ly : float
        Simulation cell dimensions in Angstrom.

    theta_max : float
        Maximum scattering angle in mrad.
        The output range is approximately [-theta_max, theta_max].

    Returns
    -------
    theta_x : ndarray
        x-axis scattering angles (mrad).

    theta_y : ndarray
        y-axis scattering angles (mrad).

    Theta_x, Theta_y : ndarray
        2D angular coordinate grids (mrad).

    Theta : ndarray
        2D total scattering angle grid (mrad).
    """

    # Physical constants
    h = 6.62607015e-34        # Planck constant (J s)
    me = 9.1093837015e-31     # electron mass (kg)
    e = 1.602176634e-19       # elementary charge (C)
    c = 299792458             # speed of light (m/s)


    # Relativistic electron wavelength
    lambda_m = h / np.sqrt(
        2 * me * e * V *
        (1 + e * V / (2 * me * c**2))
    )

    lambda_A = lambda_m * 1e10   # meter -> Angstrom

    # Electron wavevector
    k0 = 1 / lambda_A


    # Angular pixel spacing (mrad)
    dtheta_x = 1000 / (Lx * k0)
    dtheta_y = 1000 / (Ly * k0)


    # FFT index range
    nx = int(np.floor(theta_max / dtheta_x))
    ny = int(np.floor(theta_max / dtheta_y))


    # Angular coordinates
    theta_x = np.arange(-nx, nx + 1) * dtheta_x
    theta_y = np.arange(-ny, ny + 1) * dtheta_y


    # 2D angular grid
    Theta_x, Theta_y = np.meshgrid(
        theta_x,
        theta_y,
        indexing="xy"
    )

    Theta = np.sqrt(
        Theta_x**2 + Theta_y**2
    )


    return theta_x, theta_y, Theta_x, Theta_y, Theta
    
    
    ''' 
    Example
    V = 60000  # 60 keV

Lx = 39.09047055906947
Ly = 39.09515918269145

theta_x, theta_y, Theta_x, Theta_y, Theta = tacaw_theta_grid(
    V,
    Lx,
    Ly,
    theta_max=100
)

print(theta_x.shape)
print(theta_y.shape)
print(theta_x[0], theta_x[-1])
    '''
