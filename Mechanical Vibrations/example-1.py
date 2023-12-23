from Spring_Damper_Force_Sytems import spring_damper_force_system as sds
import numpy as np
import cProfile

if __name__ == '__main__':
    # Constants for the force function
    T = 1.5  # Period of the function
    N = 20   # Number of terms in the Fourier series
    w = 2 * np.pi / T  # Angular frequency

    # Constants for the system 
    m = 5.0  # mass of the beam
    g = 9.81  # gravitational acceleration
    k = 3500.0  # spring constant
    c = 270.0   # damping coefficient
    a = 0.75   # distance from pivot to spring/damper connection
    l = 2*a   # length of the beam
    lc = a   # distance from pivot to damper
    lk = a   # distance from pivot to spring
    lf = a   # distance from pivot to force
    
    
    # Force function
    def f(t, T=T):
        t = t % T  # Periodic function
        if 0 <= t <= 1:
            return 100*(2734*t**5 - 6718*t**4 + 5682*t**3 - 1931*t**2 + 228*t)
        elif 1 < t <= T:
            return -500

    # Create an instance of the spring_damper_system class
    system = sds(T, N, m, k, c, l, lk, lc, lf, f, g=g, time_span=10)

    # initial conditions
    theta_0 = 0.0  # initial angular displacement
    omega_0 = 0.0  # initial angular velocity
    
    # Solve the differential equation for the rotational motion about the end
    solution_end, solution_center, Fa_values, approximation, original, t = system.run(
        n_t=3, initial_conditions=[theta_0, omega_0])

    # Create a DataFrame with the coefficients
    df_coefs = system.create_coefficients_df()
    df_coefs.to_csv('coefficients.csv')
    print(df_coefs)

    df_freq_amp = system.create_frequency_amplitude_df()
    df_freq_amp.to_csv('frequency_amplitude.csv')
    print(df_freq_amp)
    
    print(system.create_string_representation())

    # Plot the solution
    system.plot(solution_end, solution_center,
                Fa_values, approximation, original, t)


    