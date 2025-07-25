#!/usr/bin/env python3
"""
Mock CRS Helper - Simplified mock CRS creation and configuration for testing.

This module provides utilities for creating and configuring a MockCRS instance
with simulated KID resonators. It's designed to be used by scripts that need
to test the rfmux algorithms without real hardware.

Example usage:
    from mock_crs_helper import create_mock_crs
    
    # Create with default configuration
    crs = await create_mock_crs()
    
    # Create with custom configuration
    crs = await create_mock_crs(num_resonances=20, enable_bifurcation=True)
"""

import asyncio
from typing import Optional, Dict, Any

# Import required rfmux modules
from rfmux.core.session import load_session
from rfmux.core.crs import CRS


# Default mock configuration parameters
DEFAULT_MOCK_CONFIG = {
    # Resonator generation parameters
    'num_resonances': 10,
    'freq_start': 4e9,  # 4 GHz
    'freq_end': 8e9,    # 8 GHz
    
    # Quality factor parameters
    'q_min': 1e4,
    'q_max': 1e5,
    'q_variation': 0.3,
    
    # Coupling parameters
    'coupling_min': 0.5,
    'coupling_max': 2.0,
    
    # Nonlinearity parameters
    'kinetic_inductance_fraction': 0.05,
    'kinetic_inductance_variation': 0.2,
    'frequency_shift_power_law': 0.5,
    'frequency_shift_magnitude': 0.01,
    'power_normalization': 1e-12,
    
    # Bifurcation parameters
    'enable_bifurcation': False,
    'bifurcation_iterations': 10,
    'bifurcation_convergence_tolerance': 1e-6,
    'bifurcation_damping_factor': 0.5,
    'saturation_power': 1e-13,
    'saturation_sharpness': 2.0,
    
    # Noise parameters
    'base_noise_level': 1e-4,
    'amplitude_noise_coupling': 0.01,
    'udp_noise_level': 0.1
}


async def create_mock_crs(
    module: int = 1,
    udp_host: str = '127.0.0.1',
    udp_port: int = 9876,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> CRS:
    """
    Create and configure a MockCRS instance with simulated resonators.
    
    This function:
    1. Creates a MockCRS using the proper session/flavour syntax
    2. Resolves the CRS object
    3. Configures simulated resonators with specified parameters
    4. Starts UDP streaming for data visualization
    
    Args:
        module: Module number to use (default: 1)
        udp_host: Host for UDP streaming (default: '127.0.0.1')
        udp_port: Port for UDP streaming (default: 9876)
        config: Optional custom configuration dict. If None, uses defaults.
                Keys can include any parameters from DEFAULT_MOCK_CONFIG.
        verbose: Whether to print status messages (default: True)
    
    Returns:
        CRS: A configured MockCRS instance ready for use
        
    Raises:
        Exception: If MockCRS creation or configuration fails
    """
    
    # Merge custom config with defaults
    mock_config = DEFAULT_MOCK_CONFIG.copy()
    if config:
        mock_config.update(config)
    
    if verbose:
        print("="*60)
        print("Creating Mock CRS")
        print("="*60)
        print(f"Module: {module}")
        print(f"UDP Streaming: {udp_host}:{udp_port}")
        print(f"Resonators: {mock_config['num_resonances']}")
        print(f"Frequency Range: {mock_config['freq_start']/1e9:.1f} - {mock_config['freq_end']/1e9:.1f} GHz")
        print(f"Bifurcation: {'Enabled' if mock_config['enable_bifurcation'] else 'Disabled'}")
        print("="*60)
    
    try:
        # Create MockCRS using the proper flavour syntax
        if verbose:
            print("\n1. Creating MockCRS session...")
        
        session = load_session("""
!HardwareMap
- !flavour "rfmux.core.mock"
- !CRS { serial: "MOCK0001", hostname: "127.0.0.1" }
""")
        
        # Get the CRS object
        crs = session.query(CRS).one()
        
        # Resolve the CRS (initialize connection)
        if verbose:
            print("2. Resolving MockCRS...")
        await crs.resolve()
        
        # Configure resonators
        if verbose:
            print(f"3. Generating {mock_config['num_resonances']} simulated resonators...")
        
        resonator_count = await crs.generate_resonators(mock_config)
        
        if verbose:
            print(f"   ✓ Generated {resonator_count} resonators")
        
        # Start UDP streaming
        if verbose:
            print(f"4. Starting UDP streaming on {udp_host}:{udp_port}...")
        
        await crs.start_udp_streaming(host=udp_host, port=udp_port)
        
        if verbose:
            print("   ✓ UDP streaming started")
            print("\n" + "="*60)
            print("Mock CRS ready for use!")
            print("="*60 + "\n")
        
        return crs
        
    except Exception as e:
        error_msg = f"Failed to create Mock CRS: {str(e)}"
        if verbose:
            print(f"\n❌ Error: {error_msg}")
        raise Exception(error_msg) from e


async def reconfigure_mock_crs(
    crs: CRS,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> int:
    """
    Reconfigure an existing MockCRS with new parameters.
    
    This is useful for changing resonator parameters without recreating
    the entire CRS object.
    
    Args:
        crs: An existing MockCRS instance
        config: Configuration dict with parameters to update
        verbose: Whether to print status messages
        
    Returns:
        int: Number of resonators generated
        
    Raises:
        Exception: If reconfiguration fails
    """
    if config is None:
        config = DEFAULT_MOCK_CONFIG
    
    if verbose:
        print("\nReconfiguring Mock CRS...")
        print(f"New parameters: {len(config)} settings")
    
    try:
        resonator_count = await crs.generate_resonators(config)
        
        if verbose:
            print(f"✓ Regenerated {resonator_count} resonators")
        
        return resonator_count
        
    except Exception as e:
        error_msg = f"Failed to reconfigure Mock CRS: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        raise Exception(error_msg) from e


# Convenience function for quick testing
async def test_mock_crs():
    """
    Quick test function to verify mock CRS creation.
    
    Creates a mock CRS with default settings and performs a simple
    network analysis to verify it's working.
    """
    print("Testing Mock CRS creation...")
    
    try:
        # Create mock CRS
        crs = await create_mock_crs(verbose=True)
        
        # Perform a simple test - get samples
        print("\nTesting data acquisition...")
        samples = await crs.get_samples(10, average=True, channel=1, module=1)
        
        print(f"✓ Successfully acquired {len(samples.mean.i)} samples")
        print(f"  Mean I: {samples.mean.i[0]:.3f}")
        print(f"  Mean Q: {samples.mean.q[0]:.3f}")
        
        print("\nMock CRS test completed successfully!")
        
        return crs
        
    except Exception as e:
        print(f"\n❌ Mock CRS test failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_mock_crs())
