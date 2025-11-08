import unittest
import numpy as np
from src.orbital_mechanics import calculate_acceleration, get_derivatives, runge_kutta_4
from src.constants import GM_EARTH, RADIUS_EARTH

class TestOrbitalMechanics(unittest.TestCase):
    def setUp(self):
        self.r_test = np.array([7000.0, 0.0, 0.0])
        self.v_test = np.array([0.0, 7.546, 0.0])
        self.state_test = np.hstack((self.r_test, self.v_test))
        self.delta_t = 60.0
        self.num_steps = 10

    #TESTES PARA calculate_acceleration
    def test_acceleration_magnitude(self):
        r_magnitude = np.linalg.norm(self.r_test)
        a_expected_magnitude = GM_EARTH / (r_magnitude**2)

        a_calculated_vector = calculate_acceleration(self.r_test)
        a_calculated_magnitude = np.linalg.norm(a_calculated_vector)

        self.assertAlmostEqual(a_expected_magnitude, a_calculated_magnitude, places=5,
                               msg="Acceleration magnitude does not correspond to expected (GM/r^2)")

    def test_acceleration_direction(self):
        a_calculated_vector = calculate_acceleration(self.r_test)

        self.assertTrue(a_calculated_vector[0] < 0,
                        msg="The X component from acceleration should not be negative")
        self.assertAlmostEqual(a_calculated_vector[1], 0.0, places=10,
                               msg="The Y component from acceleration should be zero")
        self.assertAlmostEqual(a_calculated_vector[2], 0.0, places=10,
                               msg="The Z component from acceleration should be zero")

    #TESTES PARA get_derivatives
    def test_derivatives_output_size(self):
        derivs = get_derivatives(0, self.state_test)
        self.assertEqual(len(derivs), 6,
                         msg="The vector of derivatives must have 6 components.")

    def test_derivatives_velocity_component(self):
        derivs = get_derivatives(0, self.state_test)
        np.testing.assert_array_almost_equal(self.v_test, derivs[0:3], decimal=5,
                                             err_msg="The first three components should be the velocities.")

    #TESTE BÁSICO PARA runge_kutta_4 (Teste de Estabilidade/Movimento)
        def test_rk4_propagation_consistency(self):
            time_series, state_history = runge_kutta_4(self.state_test, self.delta_t, self.num_steps)

            expected_final_time = self.num_steps * self.delta_t
            self.assertAlmostEqual(time_series[-1], expected_final_time, places=5,
                                   msg="The final state of the simulation it is incorrect.")

            final_position = state_history[-1, 0:3]
            initial_position = state_history[0, 0:3]

            position_difference = np.linalg.norm(final_position - initial_position)

            self.assertTrue(position_difference > 10.0,
                            msg="The final state does not change significatively after the propagation.")

if __name__ == '__main__':
    unittest.main()