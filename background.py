import numpy as np

# a simple function to convert complex numbers from cartesian to polar notation
def cart2pol(x, y):
    rho = np.sqrt((x**2 + y**2))
    phi = np.arctan2(y, x)
    return (rho, phi)

# a simple function to convert complex numbers from polar to cartesian notation
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

# a simple function to convert complex number from cartesian to polar notation
# the function takes the whole number as an argument and can return the angle in degrees
def cart2pol(z, deg=False):
  mag = np.abs(z)
  angle = np.angle(z, deg=deg)
  return (mag, angle)