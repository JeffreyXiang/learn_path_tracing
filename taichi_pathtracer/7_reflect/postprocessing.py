import taichi as ti
from dtypes import Mat3f


@ti.func
def ACES_tonemapping(color):
    aces_input_matrix = Mat3f([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])

    aces_output_matrix = Mat3f([
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602]
    ])
    
    v = aces_input_matrix @ color
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    v = a / b
    
    return ti.math.max(aces_output_matrix @ v, 0.0)


@ti.func
def gamma_correction(color, gamma):
    return color**(1/gamma)
