import numpy as np
from .constants import GM_EARTH

# 1. FUNÇÃO DE CÁLCULO DA ACELERAÇÃO

def calculate_acceleration(r_vector):
    r_magnitude = np.linalg.norm(r_vector)

    a_magnitude = -GM_EARTH /(r_magnitude**2)

    a_vector = a_magnitude * (r_vector / r_magnitude)

    return a_vector

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
