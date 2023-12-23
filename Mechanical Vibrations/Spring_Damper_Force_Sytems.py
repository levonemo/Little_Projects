import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


class spring_damper_force_system:
    def __init__(self, T, N, mass, k, c, l, lk, lc, lf, force_function, g=9.81, time_span=10) -> None:
        # Constants for the force function
        self.force_function = force_function
        self.T = T  # Period of the function
        self.N = N   # Number of terms in the Fourier series
        self.w = 2 * np.pi / T  # Angular frequency

        # Constants for the system (assumed values since they are not provided)
        self.m = mass  # mass of the beam
        self.g = g   # gravitational acceleration
        self.k = k   # spring constant
        self.c = c   # damping coefficient
        self.l = l   # length of the beam
        self.lk = lk  # distance from point A to spring
        self.lc = lc  # distance from point A to damper
        self.lf = lf  # distance from point A to force
        # moment of inertia of the beam about point A
        self.I_end = self.moment_of_inertia_end(self.m, self.l)

        # moment of inertia of the beam about its center of mass
        self.I_center = self.moment_of_inertia_center(self.m, self.l)
        self.time_span = [0, time_span]  # time range for the simulation
        self.a_n, self.b_n = self.calculate_coefficients(N, T, self.w)

    def moment_of_inertia_end(self, m, l):
        return (m * l ** 2) / 3

    def moment_of_inertia_center(self, m, l):
        return (m * l ** 2) / 12

    def calculate_coefficients(self, N, T, w):
        def a_n(n):
            result, _ = quad(lambda t: self.force_function(t)
                             * np.cos(n * w * t), 0, T)
            return result * 2 / T

        def b_n(n):
            result, _ = quad(lambda t: self.force_function(t)
                             * np.sin(n * w * t), 0, T)
            return result * 2 / T

        a = [a_n(n) for n in range(N + 1)]
        b = [b_n(n) for n in range(1, N + 1)]
        return a, b

    def fourier_series(self, t, N, T, w):
        a0 = self.a_n[0] / 2
        return a0 + sum(self.a_n[n] * np.cos(n * w * t) + self.b_n[n - 1] * np.sin(n * w * t) for n in range(1, N + 1))

    def create_string_representation(self):
        a0 = self.a_n[0] / 2
        string = f'{a0:.5}'
        for n in range(1, self.N + 1):
            string += f'{self.a_n[n]:.5}*cos({n}*w*t) + {self.b_n[n-1]:.5}*sin({n}*w*t) + '
        return string[:-2]

    def create_frequency_amplitude_df(self):
        # Calculate amplitude for each frequency component
        amplitudes = [(np.sqrt((self.a_n[i])**2 + self.b_n[i-1]**2))
                      for i in range(1, self.N + 1)]
        frequencies = [(i * self.w) for i in range(1, self.N + 1)]

        # Create a DataFrame
        df = pd.DataFrame({
            'Frequency (w)': frequencies,
            'Amplitude': amplitudes,
        })

        return df

    def rotational_motion_end(self, t, y):
        theta, omega = y  # y contains theta and omega (angular velocity)
        dtheta_dt = omega
        domega_dt = (self.fourier_series(t, self.N, self.T, self.w) * self.lf -
                     self.k * self.lk**2 * theta - self.c * self.lc**2 * omega) / self.I_end
        return [dtheta_dt, domega_dt]

    # Define the differential equation for the rotational motion about the center
    def rotational_motion_center(self, t, y):
        theta, omega = y  # y contains theta and omega (angular velocity)
        dtheta_dt = omega
        domega_dt = (self.fourier_series(t, self.N, self.T, self.w) * self.lf - self.c * self.l * self.l /
                     2 * omega - self.k * self.lk * self.l/2 * theta) / (self.I_center + (self.m * self.l**2 / 4))
        return [dtheta_dt, domega_dt]

    # Calculate the support force Fa at any time t
    def calculate_Fa(self, t, theta, omega):
        domega_dt = (self.fourier_series(t, self.N, self.T, self.w) * self.lf - self.c * self.l * self.l /
                     2 * omega - self.k * self.lk * self.l/2 * theta) / (self.I_center + (self.m * self.l**2 / 4))
        # Get the current value of the external force
        F_A = self.fourier_series(
            t, self.N, self.T, self.w) * (self.lf - self.l/2) - self.I_center * domega_dt
        return F_A

    def run(self, n_t=3, initial_conditions=[0, 0]):

        # Solve the differential equations using solve_ivp
        solution_end = solve_ivp(self.rotational_motion_end, self.time_span, initial_conditions,
                                 t_eval=np.linspace(self.time_span[0], self.time_span[1], 300))

        solution_center = solve_ivp(self.rotational_motion_center, self.time_span,
                                    initial_conditions, t_eval=np.linspace(self.time_span[0], self.time_span[1], 300))

        # Calculate support force Fa for each time step
        Fa_values = [self.calculate_Fa(t, theta, omega) for t, theta, omega in zip(
            solution_center.t, solution_center.y[0], solution_center.y[1])]
        
        print("Max Fa force is: {}N".format(max(Fa_values)))

        t = np.linspace(0, n_t*self.T, 400)
        original = np.array([self.force_function(ti) for ti in t])
        approximation = np.array(
            [self.fourier_series(ti, self.N, self.T, self.w) for ti in t])

        return solution_end, solution_center, Fa_values, approximation, original, t

    def plot(self, solution_end, solution_center, Fa_values, approximation, original, t):
        # Plot for angular displacement
        plt.figure(figsize=(13, 6))
        plt.plot(solution_end.t, solution_end.y[0], label='${\Theta}$(t)')
        # plt.title('Angular Displacement')
        # plt.xlabel('Time (s)')
        plt.title('Açısal Değişim')
        plt.xlabel('Zaman (s)')
        plt.ylabel(r"${\Theta}$ [rad]")
        plt.legend()
        plt.grid(True)

        # Plot for support force - center
        plt.figure(figsize=(13, 6))
        # plt.plot(solution_center.t, Fa_values, label='Support Force Fa(t)')
        # plt.title('Support Force')
        # plt.xlabel('Time (s)')
        plt.plot(solution_center.t, Fa_values, label='Mesnet Tepki Kuvveti Fa(t)')
        plt.plot()
        plt.title('Mesnet Tepki')
        plt.xlabel('Zaman (s)')
        plt.ylabel('Fa [N]')
        plt.legend()
        plt.grid(True)

        # Plot for original force function vs. Fourier series approximation
        plt.figure(figsize=(13, 6))
        # plt.plot(t, original, label='Original Force Function')
        plt.plot(t, original, label='Orijinal Kuvvet Fonksiyonu')
        plt.plot(t, approximation,
                 label=f'Fourier Series Approx. ({self.N} terms)')
        # plt.title(
        #     'Original Force Function vs. Fourier Series Approximation ({} terms)'.format(self.N))
        # plt.xlabel('Time (t) [s]')
        plt.title(
            'Orijinal Kuvvet Fonksiyonu vs. Fourier Serisi Yaklaşımı ({} terim)'.format(self.N))
        plt.xlabel('Zaman (t) [s]')
        plt.ylabel('F [N]')
        plt.legend()
        plt.grid(True)

        # ============================================
        # Plot for frequency vs amplitude

        df = self.create_frequency_amplitude_df()
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(df['Frequency (w)'], df['Amplitude'])
        # plt.title('Frequency vs Amplitude')
        # plt.xlabel('Frequency (rad/s)')
        # plt.ylabel('Amplitude')
        plt.title('Frekans vs Genlik')
        plt.xlabel('Frekans (rad/s)')
        plt.ylabel('Genlik')
        plt.grid(True)
        plt.show()

        # Show all plots
        plt.show()

    def create_coefficients_df(self):
        a_n = self.a_n
        b_n = self.b_n
        b_n.insert(0, 0.0)

        df = pd.DataFrame({
            'a_n': a_n,
            'b_n': b_n,
        })

        return df
