# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:56:32 2025

@author: ricar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import c, epsilon_0, hbar
import matplotlib.gridspec as gridspec

class SPDCSpectralDensity:
    def __init__(self, L0=7.5e-3, G0=8.96e-6, T=25, w0=23.27e-6, lambda_p=532e-9):
        """
        Initialize parameters for SPDC spectral density calculation
        
        Parameters:
        -----------
        L0 : float
            Crystal length at 25°C in meters
        G0 : float
            Poling period at 25°C in meters
        T : float
            Crystal temperature in °C
        w0 : float
            Pump beam waist in meters
        lambda_p : float
            Pump wavelength in meters
        """
        self.L0 = L0
        self.G0 = G0
        self.T = T
        self.w0 = w0
        self.lambda_p = lambda_p
        self.omega_p = 2 * np.pi * c / lambda_p

        # Thermal expansion coefficients for KTP from the paper
        self.alpha = 6.7e-6  # approximate value
        self.beta = 11e-9    # approximate value
        
        # Calculate temperature-dependent parameters
        self.G = self._calculate_G()
        self.L = self._calculate_L()

    def _calculate_G(self):
        """Calculate temperature-dependent poling period"""
        return self.G0 * (1 + self.alpha * (self.T - 25) + self.beta * (self.T - 25)**2)
    
    def _calculate_L(self):
        """Calculate temperature-dependent crystal length"""
        return self.L0 * (1 + self.alpha * (self.T - 25) + self.beta * (self.T - 25)**2)

    def n_KTP(self, wavelength, T):
        """
        Sellmeier equation for the refractive index of KTP along z-axis (type-0)
        
        Parameters:
        -----------
        wavelength : float or array
            Wavelength in meters
        T : float
            Temperature in °C
            
        Returns:
        --------
        n : float or array
            Refractive index
        """
        # Convert wavelength to μm for the Sellmeier equation
        wl_um = wavelength * 1e6
        
        # Temperature dependence coefficient (approximate)
        #dn_dT = 1.5e-5  # per °C
        dT = T - 25  # difference from reference temperature
        
        # Sellmeier coefficients for KTP along z-axis (approximate values)
        A = 2.12725
        B = 1.18431
        C = 0.0514852
        D = 0.6603
        E = 100.00507
        F = 9.68956e-3
        
        a0 = 9.9587e-6 
        a1 = 9.9228e-6 
        a2 = -8.9603e-6
        a3 = 4.1010e-6
        b0 = -1.1882e-8
        b1 = 10.459e-8
        b2 = -9.8136e-8
        b3 =3.1481e-8
        
        n_squared = A + (B / (1 - (C / wl_um**2))) +(D / (1 - (E/ wl_um**2))) -(F * wl_um**2)
        n_1 = a0 + a1/wl_um + a2/(wl_um**2) + a3/(wl_um**3)
        n_2 = b0 + b1/wl_um + b2/(wl_um**2) + b3/(wl_um**3)
        n = np.sqrt(n_squared) + n_1 * dT + n_2 * dT**2
        
        return n

    def calculate_phase_mismatch(self, q_i, omega_s, q_s):
        """
        Calculate the phase mismatch as in Eq. (3)
        
        Parameters:
        -----------
        q_i : array
            Transverse momentum of idler photon in m^-1
        omega_s : float
            Angular frequency of signal photon in rad/s
        q_s : array
            Transverse momentum of signal photon in m^-1
            
        Returns:
        --------
        delta_k_z : array
            Phase mismatch in m^-1
        """
        omega_i = self.omega_p - omega_s
        
        # Calculate wavelengths
        lambda_i = 2 * np.pi * c / omega_i
        lambda_s = 2 * np.pi * c / omega_s
        lambda_p = 2 * np.pi * c / self.omega_p
        
        # Calculate refractive indices
        n_i = self.n_KTP(lambda_i, self.T)
        n_s = self.n_KTP(lambda_s, self.T)
        n_p = self.n_KTP(lambda_p, self.T)
        
        # Calculate k_z components
        k_i_z = np.sqrt((omega_i * n_i / c)**2 - q_i**2)
        k_s_z = np.sqrt((omega_s * n_s / c)**2 - q_s**2)
        k_p_z = np.sqrt((self.omega_p * n_p / c)**2 - (q_i + q_s)**2)
        
        delta_k_z = k_i_z + k_s_z - k_p_z
        
        return delta_k_z

    def pump_field(self, q_sum):
        """
        Gaussian pump beam field distribution as in Eq. (9)
        
        Parameters:
        -----------
        q_sum : array
            Sum of transverse momenta (q_i + q_s) in m^-1
            
        Returns:
        --------
        E_p : array
            Pump field amplitude
        """
        return np.exp(-(self.w0**2 * q_sum**2) / 4)

    def lambda_function(self, q_i, omega_s, q_s):
        """
        Calculate the spectral amplitude function Λ as in Eq. (2)
        
        Parameters:
        -----------
        q_i : array
            Transverse momentum of idler photon in m^-1
        omega_s : float
            Angular frequency of signal photon in rad/s
        q_s : array
            Transverse momentum of signal photon in m^-1
            
        Returns:
        --------
        Lambda : array
            Spectral amplitude
        """
        omega_i = self.omega_p - omega_s
        
        # Calculate wavelengths for refractive indices
        lambda_i = 2 * np.pi * c / omega_i
        lambda_s = 2 * np.pi * c / omega_s
        
        # Calculate refractive indices
        n_i = self.n_KTP(lambda_i, self.T)
        n_s = self.n_KTP(lambda_s, self.T)
        
        # Calculate phase mismatch
        delta_k_z = self.calculate_phase_mismatch(q_i, omega_s, q_s)
        
        # Calculate pump field
        q_sum = q_i + q_s
        E_p = self.pump_field(q_sum)
        
        # Calculate sinc term
        phase_term = (delta_k_z + 2 * np.pi / self.G) * self.L / 2
        sinc_term = np.sinc(phase_term / np.pi)
        
        # Constants term (proportional to)
        const_term = 1.0 / (n_i * n_s)*self.L*np.sqrt(omega_i*omega_s)
        
        # Spectral amplitude
        Lambda = const_term * E_p * sinc_term
        
        return Lambda

    def spectral_density(self, q_s_x, omega_s, q_i_range=None):
        """
        Calculate the spectral density S(q_s, ω_s) as in Eq. (8)
        
        Parameters:
        -----------
        q_s_x : float or array
            x-component of signal photon transverse momentum in m^-1
        omega_s : float or array
            Angular frequency of signal photon in rad/s
        q_i_range : tuple
            Range of q_i values to integrate over (min, max, num_points)
            
        Returns:
        --------
        S : float or array
            Spectral density
        """
        if q_i_range is None:
            # Default range for q_i integration
            q_i_min = -2e6
            q_i_max = 2e6
            q_i_num = 1000
        else:
            q_i_min, q_i_max, q_i_num = q_i_range
            
        # Create grid for q_i
        q_i = np.linspace(q_i_min, q_i_max, q_i_num)
        
        # For scalar inputs
        if np.isscalar(q_s_x) and np.isscalar(omega_s):
            # Assuming cylindrical symmetry, only consider x-component
            q_s = q_s_x
            
            # Calculate lambda for each q_i and square it
            lambda_squared = np.abs(self.lambda_function(q_i, omega_s, q_s))**2
            
            # Integrate over q_i
            S = np.trapz(lambda_squared, q_i)
            
            return S
        
        # For array inputs - calculate on a grid
        S = np.zeros((len(q_s_x), len(omega_s)))
        
        for i, q_s_x_val in enumerate(q_s_x):
            q_s = q_s_x_val # Only x-component
            for j, omega_s_val in enumerate(omega_s):
                lambda_squared = np.abs(self.lambda_function(q_i, omega_s_val, q_s))**2
                S[i, j] = np.trapz(lambda_squared, q_i)
        
        return S

    def calculate_spectral_density_map(self, q_s_x_range, omega_s_range, q_i_range=None):
        """
        Calculate the spectral density map S(q_s_x, ω_s)
        
        Parameters:
        -----------
        q_s_x_range : tuple
            Range of q_s_x values (min, max, num_points)
        omega_s_range : tuple
            Range of omega_s values (min, max, num_points)
        q_i_range : tuple
            Range of q_i values to integrate over (min, max, num_points)
            
        Returns:
        --------
        q_s_x_values : array
            Array of q_s_x values
        omega_s_values : array
            Array of omega_s values
        S_map : 2D array
            Spectral density map
        """
        # Create arrays for q_s_x and omega_s
        q_s_x_min, q_s_x_max, q_s_x_num = q_s_x_range
        omega_s_min, omega_s_max, omega_s_num = omega_s_range
        
        q_s_x_values = np.linspace(q_s_x_min, q_s_x_max, q_s_x_num)
        omega_s_values = np.linspace(omega_s_min, omega_s_max, omega_s_num)
        
        # Create meshgrid for vectorized calculation
        q_s_x_mesh, omega_s_mesh = np.meshgrid(q_s_x_values, omega_s_values, indexing='ij')
        
        # Calculate spectral density
        S_map = self.spectral_density(q_s_x_values, omega_s_values, q_i_range)
        
        return q_s_x_values, omega_s_values, S_map

    def plot_spectral_density(self, q_s_x_range, omega_s_range, q_i_range=None, 
                             normalize=True, cmap='viridis', log_scale=False,
                             title=None, filename=None):
        """
        Plot the spectral density map S(q_s_x, ω_s)
        
        Parameters:
        -----------
        q_s_x_range : tuple
            Range of q_s_x values (min, max, num_points)
        omega_s_range : tuple
            Range of omega_s values (min, max, num_points)
        q_i_range : tuple
            Range of q_i values to integrate over (min, max, num_points)
        normalize : bool
            Whether to normalize the spectral density
        cmap : str
            Colormap to use
        log_scale : bool
            Whether to use log scale for color
        title : str
            Title for the plot
        filename : str
            Filename to save the plot
        """
        # Calculate spectral density map
        q_s_x_values, omega_s_values, S_map = self.calculate_spectral_density_map(
            q_s_x_range, omega_s_range, q_i_range
        )
        
        # Create figure with subplots
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
        
        # Main spectral density plot
        ax_main = plt.subplot(gs[1, 0])
        
        # Normalize if requested
        if normalize:
            S_map = S_map / np.max(S_map)
        
        # Plot as image
        if log_scale and np.any(S_map > 0):
            # Add small value to avoid log(0)
            vmin = np.max(S_map) * 1e-3
            im = ax_main.pcolormesh(
                omega_s_values / (self.omega_p/2),  # Normalize to ω_p/2
                q_s_x_values * 1e-6,  # Convert to 1/μm
                np.maximum(S_map, vmin),
                norm=LogNorm(vmin=vmin),
                cmap=cmap,
                shading='auto'
            )
        else:
            im = ax_main.pcolormesh(
                omega_s_values / (self.omega_p/2),  # Normalize to ω_p/2
                q_s_x_values * 1e-6,  # Convert to 1/μm
                S_map,
                cmap=cmap,
                shading='auto'
            )
        
        # Axis labels and title
        ax_main.set_xlabel(r'$\omega_s/(\omega_p/2)$')
        ax_main.set_ylabel(r'$q_{s,x}$ [1/μm]')
        
        if title:
            plt.suptitle(title)
        else:
            params_str = f"$G_0$={self.G0*1e6:.3f} μm, $T$={self.T}°C, $w_0$={self.w0*1e6:.2f} μm, $L_0$={self.L0*1e3:.1f} mm"
            plt.suptitle(f"SPDC Spectral Density - {params_str}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main)
        cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
        
        # Top projection plot (integrated over q_s_x)
        ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
        S_integrated_q = np.sum(S_map, axis=0)
        if normalize:
            S_integrated_q = S_integrated_q / np.max(S_integrated_q)
        ax_top.plot(omega_s_values / (self.omega_p/2), S_integrated_q)
        ax_top.set_ylabel('$S(\omega_s)$')
        ax_top.set_title('Integrated over $q_{s,x}$')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        
        # Set x-range to match paper figures
        ax_main.set_xlim(0.8, 1.2)
        ax_main.set_ylim(-1.0, 1.0)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_spectral_density2(self, q_s_x_range, omega_s_range, q_i_range=None, 
                                 normalize=True, cmap='viridis', log_scale=False,
                                 title=None, filename=None):
            """
            Plot the spectral density map S(q_s_x, ω_s)
            
            Parameters:
            -----------
            q_s_x_range : tuple
                Range of q_s_x values (min, max, num_points)
            omega_s_range : tuple
                Range of omega_s values (min, max, num_points)
            q_i_range : tuple
                Range of q_i values to integrate over (min, max, num_points)
            normalize : bool
                Whether to normalize the spectral density
            cmap : str
                Colormap to use
            log_scale : bool
                Whether to use log scale for color
            title : str
                Title for the plot
            filename : str
                Filename to save the plot
            """
            # Calculate spectral density map
            q_s_x_values, omega_s_values, S_map = self.calculate_spectral_density_map(
                q_s_x_range, omega_s_range, q_i_range
            )
            
            # Create figure with subplots
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
            
            # Main spectral density plot
            ax_main = plt.subplot(gs[1, 0])
            
            # Normalize if requested
            if normalize:
                S_map = S_map / np.max(S_map)
            
            # Plot as image
            if log_scale and np.any(S_map > 0):
                # Add small value to avoid log(0)
                vmin = np.max(S_map) * 1e-3
                im = ax_main.pcolormesh(
                    omega_s_values / (self.omega_p/2),  # Normalize to ω_p/2
                    q_s_x_values * 1e-6,  # Convert to 1/μm
                    np.maximum(S_map, vmin),
                    norm=LogNorm(vmin=vmin),
                    cmap=cmap,
                    shading='auto'
                )
            else:
                im = ax_main.pcolormesh(
                    omega_s_values / (self.omega_p/2),  # Normalize to ω_p/2
                    q_s_x_values * 1e-6,  # Convert to 1/μm
                    S_map,
                    cmap=cmap,
                    shading='auto'
                )
            
            # Axis labels and title
            ax_main.set_xlabel(r'$\omega_s/(\omega_p/2)$')
            ax_main.set_ylabel(r'$q_{s,x}$ [1/μm]')
            
            if title:
                plt.suptitle(title)
            else:
                params_str = f"$G_0$={self.G0*1e6:.3f} μm, $T$={self.T}°C, $w_0$={self.w0*1e6:.2f} μm, $L_0$={self.L0*1e3:.1f} mm"
                plt.suptitle(f"SPDC Spectral Density - {params_str}")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_main)
            cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
            
            # Top projection plot (integrated over w_s)
            ax_top = plt.subplot(gs[0, 0])
            S_integrated_q = np.sum(S_map, axis=1)
            if normalize:
                S_integrated_q = S_integrated_q / np.max(S_integrated_q)
            ax_top.plot(q_s_x_values*1e-6, S_integrated_q)
            ax_top.set_ylabel('$S(q_{s,x})$')
            ax_top.set_xlabel(r'$q_{s,x}$ [1/μm]')
            ax_top.set_title('Integrated over $w_{s}$')
            plt.setp(ax_top.get_xticklabels(), visible=True)
            
            # Set x-range to match paper figures
            ax_main.set_xlim(0.8, 1.2)
            ax_main.set_ylim(-1.0, 1.0)
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                
            plt.show()
#Function to define plot the emission angle

def emission_angle(self, q_s_x_range, omega_s_range, q_i_range=None, 
                         normalize=True, cmap='viridis', log_scale=False,
                         title=None, filename=None):
    # Calculate spectral density map
    q_s_x_values, omega_s_values, S_map = self.calculate_spectral_density_map(
        q_s_x_range, omega_s_range, q_i_range
    )
    
    
    return 

# Example usage to reproduce Figure 3
def reproduce_figure3():
    # Common parameters
    L0 = 7.5e-3  # 7.5 mm
    T = 25  # 25°C
    w0 = 23.27e-6  # 23.27 μm
    lambda_p = 532e-9  # 532 nm
    
    # Create a figure similar to Figure 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define different poling periods
    G0_values = [8.95e-6, 8.96e-6, 8.97e-6]  # in meters
    
    # Range for calculation
    q_s_x_range = (-1e6, 1e6, 100)  # in m^-1
    omega_s_range = (0.8 * (2*np.pi*c/lambda_p)/2, 1.2 * (2*np.pi*c/lambda_p)/2, 100)  # in rad/s
    q_i_range = (-2e6, 2e6, 250)  # in m^-1
    
    for i, G0 in enumerate(G0_values):
        # Create SPDC model
        spdc = SPDCSpectralDensity(L0=L0, G0=G0, T=T, w0=w0, lambda_p=lambda_p)
        
        # Calculate spectral density
        q_s_x_values, omega_s_values, S_map = spdc.calculate_spectral_density_map(
            q_s_x_range, omega_s_range, q_i_range
        )
        
        # Normalize
        if i == 0:  # Normalize to the maximum of the second plot (as in the paper)
            max_val = np.max(S_map)
        S_map = S_map / max_val
        
        # Plot
        im = axes[i].pcolormesh(
            omega_s_values / (spdc.omega_p/2),
            q_s_x_values * 1e-6,
            S_map,
            cmap='hot',
            vmin=0,
            vmax=1,
            shading='auto'
        )
        
        # Axis labels
        axes[i].set_xlabel(r'$\omega_s/(\omega_p/2)$')
        if i == 0:
            axes[i].set_ylabel(r'$q_{s,x}$ [1/μm]')
        
        # Title
        axes[i].set_title(f"$G_0$ = {G0*1e6:.2f} μm")
        
        # Set axis limits to match paper
        axes[i].set_xlim(0.8, 1.2)
        axes[i].set_ylim(-1.0, 1.0)
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=1)
    cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
    
    plt.suptitle("Reproduction of Figure 3: Varying poling period $G_0$", y=1.02)
    plt.show()

# Function to reproduce other figures
def reproduce_figure4():
    """Reproduce Figure 4: Effect of beam waist w0"""
    # Common parameters
    L0 = 7.5e-3  # 7.5 mm
    T = 25  # 25°C
    G0 = 8.96e-6  # 9.018 μm
    lambda_p = 532e-9  # 532 nm
    
    # Range for calculation
    q_s_x_range = (-1e6, 1e6, 100)  # in m^-1
    omega_s_range = (0.8 * (2*np.pi*c/lambda_p)/2, 1.2 * (2*np.pi*c/lambda_p)/2, 100)  # in rad/s
    q_i_range = (-2e6, 2e6, 250)  # in m^-1
    
    # Create a figure similar to Figure 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Two different beam waists
    w0_values = [23.27e-6, 46.53e-6]  # in meters
    
    for i, w0 in enumerate(w0_values):
        # Create SPDC model
        spdc = SPDCSpectralDensity(L0=L0, G0=G0, T=T, w0=w0, lambda_p=lambda_p)
        
        # Calculate spectral density
        q_s_x_values, omega_s_values, S_map = spdc.calculate_spectral_density_map(
            q_s_x_range, omega_s_range, q_i_range
        )
        
        # Normalize
        S_map = S_map / np.max(S_map)
        
        # Plot
        im = axes[i].pcolormesh(
            omega_s_values / (spdc.omega_p/2),
            q_s_x_values * 1e-6,
            S_map,
            cmap='hot',
            vmin=0,
            vmax=1,
            shading='auto'
        )
        
        # Axis labels
        axes[i].set_xlabel(r'$\omega_s/(\omega_p/2)$')
        if i == 0:
            axes[i].set_ylabel(r'$q_{s,x}$ [1/μm]')
        
        # Title
        axes[i].set_title(f"$w_0$ = {w0*1e6:.2f} μm")
        
        # Set axis limits to match paper
        axes[i].set_xlim(0.8, 1.2)
        axes[i].set_ylim(-1.0, 1.0)
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
    
   
    plt.suptitle("Reproduction of Figure 4: Varying beam waist $w_0$", y=1.02)
    plt.show()

def reproduce_figure5():
    """Reproduce Figure 5: Effect of temperature T"""
    # Common parameters
    L0 = 15e-3  # 7.5 mm
    G0 = 3.425e-6  # 9.018 μm
    w0 = 46.53e-6  # 46.53 μm
    lambda_p = 405e-9  # 532 nm
    
    # Range for calculation
    q_s_x_range = (-1e6, 1e6, 100)  # in m^-1
    omega_s_range = (0.8 * (2*np.pi*c/lambda_p)/2, 1.2 * (2*np.pi*c/lambda_p)/2, 100)  # in rad/s
    q_i_range = (-2e6, 2e6, 250)  # in m^-1
    
    # Create a figure similar to Figure 5
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Three different temperatures
    T_values = [15, 25, 35]  # in °C
    
    for i, T in enumerate(T_values):
        # Create SPDC model
        spdc = SPDCSpectralDensity(L0=L0, G0=G0, T=T, w0=w0, lambda_p=lambda_p)
        
        # Calculate spectral density
        q_s_x_values, omega_s_values, S_map = spdc.calculate_spectral_density_map(
            q_s_x_range, omega_s_range, q_i_range
        )
        
        # Normalize
        S_map = S_map / np.max(S_map)
        
        # Plot
        im = axes[i].pcolormesh(
            omega_s_values / (spdc.omega_p/2),
            q_s_x_values * 1e-6,
            S_map,
            cmap='hot',
            vmin=0,
            vmax=1,
            shading='auto'
        )
        
        # Axis labels
        axes[i].set_xlabel(r'$\omega_s/(\omega_p/2)$')
        if i == 0:
            axes[i].set_ylabel(r'$q_{s,x}$ [1/μm]')
        
        # Title
        axes[i].set_title(f"$T$ = {T}°C")
        
        # Set axis limits to match paper
        axes[i].set_xlim(0.8, 1.2)
        axes[i].set_ylim(-1.0, 1.0)
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
    
    
    plt.suptitle("Reproduction of Figure 5: Varying temperature $T$", y=1.02)
    plt.show()

def reproduce_figure6():
    """Reproduce Figure 6: Effect of crystal length L0"""
    # Common parameters
    G0 = 8.96e-6  # 9.018 μm
    T = 25  # 25°C
    w0 = 46.53e-6  # 46.53 μm
    lambda_p = 532e-9  # 532 nm
    
    # Range for calculation
    q_s_x_range = (-1e6, 1e6, 100)  # in m^-1
    omega_s_range = (0.8 * (2*np.pi*c/lambda_p)/2, 1.2 * (2*np.pi*c/lambda_p)/2, 100)  # in rad/s
    q_i_range = (-2e6, 2e6, 250)  # in m^-1
    
    # Create a figure similar to Figure 6
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Two different crystal lengths
    L0_values = [7.5e-3, 12e-3]  # in meters
    
    for i, L0 in enumerate(L0_values):
        # Create SPDC model
        spdc = SPDCSpectralDensity(L0=L0, G0=G0, T=T, w0=w0, lambda_p=lambda_p)
        
        # Calculate spectral density
        q_s_x_values, omega_s_values, S_map = spdc.calculate_spectral_density_map(
            q_s_x_range, omega_s_range, q_i_range
        )
        
        # Normalize
        S_map = S_map / np.max(S_map)
        
        # Plot
        im = axes[i].pcolormesh(
            omega_s_values / (spdc.omega_p/2),
            q_s_x_values * 1e-6,
            S_map,
            cmap='hot',
            vmin=0,
            vmax=1,
            shading='auto'
        )
        
        # Axis labels
        axes[i].set_xlabel(r'$\omega_s/(\omega_p/2)$')
        if i == 0:
            axes[i].set_ylabel(r'$q_{s,x}$ [1/μm]')
        
        # Title
        axes[i].set_title(f"$L_0$ = {L0*1e3:.1f} mm")
        
        # Set axis limits to match paper
        axes[i].set_xlim(0.8, 1.2)
        axes[i].set_ylim(-1.0, 1.0)
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label('$S(q_{s,x}, \omega_s)$ [a.u.]')
    
    
    plt.suptitle("Reproduction of Figure 6: Varying crystal length $L_0$", y=1.02)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create model with default parameters
    # Common parameters   
    L0 = 10e-3  # 7.5 mm
    G0 = 3.428e-6  # 9.018 μm
    w0 = 95e-6  # 46.53 μm
    T = 30 # 25°C
    lambda_p = 405.02e-9  # 532 nm
    spdc = SPDCSpectralDensity(L0=L0, G0=G0, T=T, w0=w0, lambda_p=lambda_p)
    
    # Define ranges for spectral density map
    q_s_x_range = (-1e6, 1e6, 200)  # in m^-1
    omega_p = 2 * np.pi * c / lambda_p  # pump angular frequency
    omega_s_range = (0.8 * omega_p/2, 1.2 * omega_p/2, 200)  # in rad/s
    q_i_range = (-1e6, 1e6, 1000)  # in m^-1
    
    # # Plot spectral density map
    spdc.plot_spectral_density(q_s_x_range, omega_s_range, q_i_range, cmap='hot')
    spdc.plot_spectral_density2(q_s_x_range, omega_s_range, q_i_range, cmap='hot')
    
    # Reproduce figures from the paper
    reproduce_figure3()
    reproduce_figure4()
    reproduce_figure5()
    reproduce_figure6()