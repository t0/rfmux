from importlib import util
from pathlib import Path
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# point to the repo root if your test lives elsewhere
demo_path = Path(__file__).resolve().parents[1] / "home" / "Demos" / "simplified_tuning_flow.py"

spec = util.spec_from_file_location("simplified_tuning_flow", demo_path)
simplified_tuning_flow = util.module_from_spec(spec)
spec.loader.exec_module(simplified_tuning_flow)

main = simplified_tuning_flow.main
run_algorithm_flow = simplified_tuning_flow.run_algorithm_flow
sys.modules["simplified_tuning_flow"] = simplified_tuning_flow

class TestIntegration:
    """Full integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mock_mode_execution(self):
        """Test that mock mode executes without errors."""
        print("\n=== Testing mock mode execution ===")
        exit_code = await main(serial="MOCK")
        assert exit_code == 0, "Mock mode should complete successfully"
        print("Mock mode completed successfully")

class TestSimpleTuningFlow:
    """Test flow for simplified_tuning_flow.py"""

    @pytest.mark.asyncio
    async def test_network_analysis_step(self):
        """Test the network analysis step in isolation."""
        print("\n=== Testing network analysis step ===")
        
        # Create a minimal mock CRS
        mock_crs = AsyncMock()
        mock_crs.serial = "0000"
        
        # Mock the take_netanal method
        frequencies = np.linspace(100e6, 2450e6, 1000)
        iq_complex = np.random.randn(1000) + 1j * np.random.randn(1000)
        phase_degrees = np.random.randn(1000) * 180
        
        mock_crs.take_netanal = AsyncMock(return_value={
            'frequencies': frequencies,
            'iq_complex': iq_complex,
            'phase_degrees': phase_degrees
        })
        mock_crs.get_cable_length = AsyncMock(return_value=0.0)
        mock_crs.set_cable_length = AsyncMock()
        mock_crs.clear_channels = AsyncMock()
        mock_crs.py_get_samples = AsyncMock()
        
        # Mock find_resonances to return no resonances (stops flow early)
        with patch('simplified_tuning_flow.find_resonances') as mock_find:
            mock_find.return_value = {
                'resonance_frequencies': [],
                'resonances_details': []
            }
            
            # Run the flow
            MODULE = 1
            NETANAL_PARAMS = {
                'amp': 0.001,
                'fmin': 100e6,
                'fmax': 2450e6,
                'nsamps': 10,
                'npoints': 1000,
                'max_chans': 1023,
                'max_span': 500e6,
                'module': MODULE
            }
            
            await run_algorithm_flow(
                mock_crs, MODULE, NETANAL_PARAMS, {}, {}, {}, {}, full_run = False
            )
        
        # Verify network analysis was called
        assert mock_crs.take_netanal.called, "take_netanal should be called"
        call_kwargs = mock_crs.take_netanal.call_args[1]
        assert call_kwargs['fmin'] == 100e6
        assert call_kwargs['fmax'] == 2450e6
        print("Network analysis called with correct parameters")

    @pytest.mark.asyncio
    async def test_cable_delay_correction(self):
        """Test that cable delay correction is applied."""
        print("\n=== Testing cable delay correction ===")
        
        mock_crs = AsyncMock()
        mock_crs.serial = "0000"
        
        # Create data with known cable delay
        frequencies = np.linspace(100e6, 2450e6, 100)
        tau = 10e-9  # 10 ns delay
        phase_degrees = -360 * frequencies * tau
        
        mock_crs.take_netanal = AsyncMock(return_value={
            'frequencies': frequencies,
            'iq_complex': np.exp(1j * np.deg2rad(phase_degrees)),
            'phase_degrees': phase_degrees
        })
        mock_crs.get_cable_length = AsyncMock(return_value=0.0)
        mock_crs.set_cable_length = AsyncMock()
        mock_crs.clear_channels = AsyncMock()
        mock_crs.py_get_samples = AsyncMock()
        
        with patch('simplified_tuning_flow.find_resonances') as mock_find:
            mock_find.return_value = {
                'resonance_frequencies': [],
                'resonances_details': []
            }
            
            await run_algorithm_flow(
                mock_crs, 1,
                {'amp': 0.001, 'fmin': 100e6, 'fmax': 2450e6,
                 'nsamps': 10, 'npoints': 100, 'module': 1},
                {}, {}, {}, {}, full_run = False
            )
        
        # Verify cable length was set
        assert mock_crs.set_cable_length.called, "Cable length should be set"
        call_args = mock_crs.set_cable_length.call_args
        new_length = call_args[1]['length']
        print(f"✓ Cable length set to: {new_length:.2f} m")
    
    @pytest.mark.asyncio
    async def test_resonance_finding(self):
        """Test that resonance finding works correctly."""
        print("\n=== Testing resonance finding ===")
        
        mock_crs = AsyncMock()
        mock_crs.serial = "0000"
        
        # Create data with simulated resonances
        frequencies = np.linspace(400e6, 800e6, 10000)
        
        # Add two resonances
        iq_complex = np.ones(10000, dtype=complex)
        for f0 in [500e6, 600e6]:
            Q = 2e4
            response = 1 / (1 + 2j * Q * (frequencies - f0) / f0)
            iq_complex *= response
        
        phase_degrees = np.angle(iq_complex, deg=True)
        
        mock_crs.take_netanal = AsyncMock(return_value={
            'frequencies': frequencies,
            'iq_complex': iq_complex,
            'phase_degrees': phase_degrees
        })
        mock_crs.get_cable_length = AsyncMock(return_value=0.0)
        mock_crs.set_cable_length = AsyncMock()
        mock_crs.clear_channels = AsyncMock()
        mock_crs.py_get_samples = AsyncMock()
        
        # Don't mock find_resonances - let it run
        # Mock everything after finding resonances
        mock_crs.multisweep = AsyncMock(return_value={})
        
        with patch('simplified_tuning_flow.bias_kids') as mock_bias:
            mock_bias.return_value = {}
            
            await run_algorithm_flow(
                mock_crs, 1,
                {'amp': 0.001, 'fmin': 400e6, 'fmax': 800e6,
                 'nsamps': 10, 'npoints': 10000, 'module': 1},
                {'min_dip_depth_db': 0.5, 'min_Q': 1e4, 'max_Q': 1e7,
                 'min_resonance_separation_hz': 50e3, 'data_exponent': 2.0},
                {'span_hz': 500e3, 'npoints_per_sweep': 50, 'amp': 0.001,
                 'nsamps': 10, 'module': 1},
                {'apply_skewed_fit': False, 'apply_nonlinear_fit': False},
                {'num_samples': 1000, 'module': 1}, full_run = False
            )
        
        print("Resonance finding completed")
    
    @pytest.mark.asyncio
    async def test_full_flow_with_resonances(self):
        """Test complete flow when resonances are found."""
        print("\n=== Testing full flow with resonances ===")
        
        mock_crs = AsyncMock()
        mock_crs.serial = "0000"
        
        # Setup network analysis
        frequencies = np.linspace(100e6, 2450e6, 1000)
        mock_crs.take_netanal = AsyncMock(return_value={
            'frequencies': frequencies,
            'iq_complex': np.random.randn(1000) + 1j * np.random.randn(1000),
            'phase_degrees': np.random.randn(1000) * 180
        })
        mock_crs.get_cable_length = AsyncMock(return_value=0.0)
        mock_crs.set_cable_length = AsyncMock()
        mock_crs.clear_channels = AsyncMock()
        
        # Mock resonances
        test_resonances = [500e6, 600e6, 700e6]
        
        # Mock multisweep results
        multisweep_results = {}
        for i, f in enumerate(test_resonances):
            freqs = np.linspace(f - 250e3, f + 250e3, 50)
            multisweep_results[i] = {
                'frequencies': freqs,
                'iq_complex': np.random.randn(50) + 1j * np.random.randn(50),
                'bias_frequency': f,
                'fit_params': {
                    'fr': f,
                    'Qr': 1e4,
                    'Qc': 5e4,
                    'Qi': 5e4
                }
            }
        
        mock_crs.multisweep = AsyncMock(return_value=multisweep_results)
        
        # Mock noise data
        mock_spectrum = Mock()
        mock_spectrum.psd_i = [np.random.randn(100) for _ in range(3)]
        mock_spectrum.psd_q = [np.random.randn(100) for _ in range(3)]
        mock_spectrum.freq_iq = np.linspace(0, 500, 100)
        mock_spectrum.freq_dsb = np.linspace(-500, 500, 100)
        
        mock_samples = Mock()
        mock_samples.spectrum = mock_spectrum
        mock_crs.py_get_samples = AsyncMock(return_value=mock_samples)
        
        # Patch the algorithm functions
        with patch('simplified_tuning_flow.find_resonances') as mock_find, \
             patch('simplified_tuning_flow.bias_kids') as mock_bias, \
             patch('simplified_tuning_flow.fit_skewed_multisweep') as mock_fit:
            
            # Mock find_resonances
            mock_find.return_value = {
                'resonance_frequencies': test_resonances,
                'resonances_details': [
                    {'q_estimated': 1e4, 'prominence_db': 3.0}
                    for _ in test_resonances
                ]
            }
            
            # Mock fitting
            mock_fit.return_value = multisweep_results
            
            # Mock bias
            bias_results = {}
            for i, f in enumerate(test_resonances):
                bias_results[i] = {
                    'bias_frequency': f,
                    'df_calibration': 1e6
                }
            mock_bias.return_value = bias_results
            
            # Run full flow
            await run_algorithm_flow(
                mock_crs, 1,
                {'amp': 0.001, 'fmin': 100e6, 'fmax': 2450e6,
                 'nsamps': 10, 'npoints': 1000, 'module': 1},
                {'min_dip_depth_db': 1.0, 'min_Q': 1e4, 'max_Q': 1e7,
                 'min_resonance_separation_hz': 100e3, 'data_exponent': 2.0},
                {'span_hz': 500e3, 'npoints_per_sweep': 50, 'amp': 0.001,
                 'nsamps': 10, 'module': 1, 'bias_frequency_method': 'max-diq',
                 'rotate_saved_data': False, 'sweep_direction': 'upward'},
                {'apply_skewed_fit': True, 'apply_nonlinear_fit': False,
                 'approx_Q_for_fit': 1e4, 'fit_resonances': True,
                 'center_iq_circle': True, 'normalize_fit': True},
                {'num_samples': 1000, 'return_spectrum': True,
                 'scaling': 'psd', 'reference': 'absolute',
                 'nsegments': 5, 'spectrum_cutoff': 0.9,
                 'channel': None, 'module': 1}, full_run = False
            )
        
        # Verify all major steps were called
        assert mock_crs.take_netanal.called, "Network analysis should run"
        assert mock_find.called, "Resonance finding should run"
        assert mock_crs.multisweep.called, "Multisweep should run"
        assert mock_bias.called, "Bias should run"
        assert mock_crs.py_get_samples.called, "Noise sampling should run"
        
        print("Full flow completed with all steps")
    
    @pytest.mark.asyncio
    async def test_no_resonances_handling(self):
        """Test that flow handles no resonances gracefully."""
        print("\n=== Testing no resonances case ===")
        
        mock_crs = AsyncMock()
        mock_crs.serial = "0000"
        
        # Flat response - no resonances
        frequencies = np.linspace(100e6, 2450e6, 1000)
        mock_crs.take_netanal = AsyncMock(return_value={
            'frequencies': frequencies,
            'iq_complex': np.ones(1000) + 1j * np.zeros(1000),
            'phase_degrees': np.zeros(1000)
        })
        mock_crs.get_cable_length = AsyncMock(return_value=0.0)
        mock_crs.set_cable_length = AsyncMock()
        mock_crs.clear_channels = AsyncMock()
        mock_crs.multisweep = AsyncMock()
        mock_crs.py_get_samples = AsyncMock()
        
        with patch('simplified_tuning_flow.find_resonances') as mock_find:
            mock_find.return_value = {
                'resonance_frequencies': [],
                'resonances_details': []
            }
            
            await run_algorithm_flow(
                mock_crs, 1,
                {'amp': 0.001, 'fmin': 100e6, 'fmax': 2450e6,
                 'nsamps': 10, 'npoints': 1000, 'module': 1},
                {}, {}, {}, {}, full_run = False
            )
        
        # Should not call multisweep when no resonances
        mock_crs.multisweep.assert_not_called()
        print("Flow correctly handles no resonances")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in main function."""
        print("\n=== Testing error handling ===")
        
        with patch('simplified_tuning_flow.create_mock_crs') as mock_create:
            mock_create.side_effect = Exception("Simulated error")
            
            exit_code = await main(serial="MOCK")
            assert exit_code == 1, "Should return error code on exception"
        
        print("Errors handled gracefully")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])