import numpy as np
from .constants import GM_EARTH, RADIUS_EARTH, J2

# 1. FUNÇÃO DE CÁLCULO DA ACELERAÇÃO

def calculate_acceleration(r_vector):
    r_norm = np.linalg.norm(r_vector)
    x, y, z = r_vector

    a_kepler = -GM_EARTH / (r_norm ** 3) * r_vector

    # J2_factor = (3/2) * J2 * (GM * R_Terra^2) / r^5
    k_j2 = 1.5 * J2 * GM_EARTH * (RADIUS_EARTH ** 2) / (r_norm ** 5)

    z2 = z ** 2
    r2 = r_norm ** 2

    ax_j2 = k_j2 * x * (5 * z2 / r2 - 1)
    ay_j2 = k_j2 * y * (5 * z2 / r2 - 1)
    az_j2 = k_j2 * z * (5 * z2 / r2 - 3)

    a_j2 = np.array([ax_j2, ay_j2, az_j2])

    return a_kepler + a_j2

# 2. FUNÇÃO DE RETORNO DAS VARIÁVEIS
def get_derivatives(t, state):
    r_vector = state[0:3]
    v_vector = state[3:6]

    a_vector = calculate_acceleration(r_vector)

    derivs = np.hstack((v_vector, a_vector))

    return derivs

def runge_kutta_4(state_initial, delta_t, num_steps):
    num_variables = len(state_initial)
    state_history = np.zeros((num_steps + 1, num_variables))
    time_series = np.zeros(num_steps + 1)

    state_history[0, :] = state_initial

    state_current = state_initial.copy()

    for i in range(num_steps):
        t_current = time_series[i]

        k1 = delta_t * get_derivatives(t_current, state_current)

        k2 = delta_t * get_derivatives(t_current + 0.5 * delta_t, state_current + 0.5 * k1)

        k3 = delta_t * get_derivatives(t_current + 0.5 * delta_t, state_current + 0.5 * k2)

        k4 = delta_t * get_derivatives(t_current + delta_t, state_current + k3)

        state_next = state_current + (1.0/6.0) *(k1 + 2.0*k2 + 2.0*k3 + k4)

        state_current = state_next
        state_history[i + 1, :] = state_next
        time_series[i + 1] = t_current + delta_t
    return time_series, state_history
