#!/usr/bin/env python3
"""
Mock CRS Helper - Simplified mock CRS creation and configuration for testing.

This module provides utilities for creating and configuring a MockCRS instance
with simulated KID resonators. It's designed to be used by scripts that need
to test the rfmux algorithms without real hardware.

Now uses the unified Single Source of Truth (SoT) in rfmux.mock.config.

Example usage:
    from mock_crs_helper import create_mock_crs
    
    # Create with default configuration
    crs = await create_mock_crs()
    
    # Create with custom configuration
    crs = await create_mock_crs(num_resonances=20, auto_bias_kids=True)
"""

import asyncio
from typing import Optional, Dict, Any

# Import required rfmux modules
from rfmux.core.session import load_session
from rfmux.core.crs import CRS
from rfmux.mock import config as mc


async def create_mock_crs(
    module: int = 1,
    udp_host: str | None = None,
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
        udp_host: Host for UDP streaming (default: None, meaning
            multicast if this machine can and loopback unicast if not)
        udp_port: Port for UDP streaming (default: 9876)
        config: Optional custom configuration dict. If None, uses unified defaults.
                Keys should match rfmux.mock.config.MOCK_DEFAULTS.
        verbose: Whether to print status messages (default: True)
    
    Returns:
        CRS: A configured MockCRS instance ready for use
        
    Raises:
        Exception: If MockCRS creation or configuration fails
    """
    
    # Merge custom config with SoT defaults
    merged = mc.apply_overrides(config)

    if verbose:
        print("="*60)
        print("Creating Mock CRS")
        print("="*60)
        print(f"Module: {module}")
        print(f"UDP Streaming: {udp_host or 'auto'}:{udp_port}")
        print(f"Resonators: {merged['num_resonances']}")
        print(f"Frequency Range: {merged['freq_start']/1e9:.1f} - {merged['freq_end']/1e9:.1f} GHz")
        print("="*60)
    
    try:
        # Create MockCRS using the proper flavour syntax
        if verbose:
            print("\n1. Creating MockCRS session...")

        session = load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
        
        # Get the CRS object
        crs = session.query(CRS).one()
        
        # Resolve the CRS (initialize connection)
        if verbose:
            print("2. Resolving MockCRS...")
        await crs.resolve()
        
        # Configure resonators
        if verbose:
            print(f"3. Generating {merged['num_resonances']} simulated resonators...")
        
        resonator_count = await crs.generate_resonators(merged)
        
        if verbose:
            print(f"   ✓ Generated {resonator_count} resonators")
        
        # Start UDP streaming
        if verbose:
            print(f"4. Starting UDP streaming on {udp_host or 'auto'}:{udp_port}...")
        
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
    merged = mc.apply_overrides(config or mc.defaults())
    
    if verbose:
        print("\nReconfiguring Mock CRS...")
        print(f"New parameters: {len(merged)} settings")
    
    try:
        resonator_count = await crs.generate_resonators(merged)
        
        if verbose:
            print(f"✓ Regenerated {resonator_count} resonators")
        
        return resonator_count
        
    except Exception as e:
        error_msg = f"Failed to reconfigure Mock CRS: {str(e)}"
        if verbose:
            print(f"❌ Error: {error_msg}")
        raise Exception(error_msg) from e


# ── Bringing a running mock to a new configuration ────────────────

_PULSE_PREFIX = "pulse_"


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)
    return a == b


def config_changes(previous: Dict[str, Any], config: Dict[str, Any]) -> set:
    """Keys *config* changes against *previous*.

    A key *config* lacks, or carries as None, is not a change: the
    dialog only edits what it shows and hands back None for the rest.
    Floats that have been through a text field compare with tolerance.
    """
    return {k for k, v in config.items()
            if v is not None and not _same(previous.get(k), v)}


def merged(previous: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """*previous* with *config*'s non-None values on top."""
    out = dict(previous)
    out.update({k: v for k, v in config.items() if v is not None})
    return out


def pulse_only_change(changed) -> bool:
    """True when every changed key is a pulse setting, which the running
    model takes live through set_pulse_mode; anything else means the
    array has to be regenerated.  An empty change is pulse-only too."""
    return all(k.startswith(_PULSE_PREFIX) for k in changed)


def pulse_mode_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """set_pulse_mode's keyword arguments from a mock config: the
    pulse_* keys without their prefix, the mode aside."""
    return {k[len(_PULSE_PREFIX):]: v for k, v in config.items()
            if k.startswith(_PULSE_PREFIX) and k != "pulse_mode"}


async def apply_mock_config(crs: CRS, config: Dict[str, Any],
                            previous: Optional[Dict[str, Any]] = None):
    """Bring a running mock to *config*.

    Pulse settings are taken live through set_pulse_mode.  Anything
    else regenerates the array, which at many tones is seconds during
    which the stream stalls, so only what changed against *previous*
    decides; with no *previous*, everything counts as changed.  A
    regeneration pins a random seed into *config* first, so the client
    and the server agree on the array that was built.

    Returns ``(outcome, resonator_count)`` with outcome one of
    ``"unchanged"``, ``"pulses"``, ``"regenerated"``; the count is None
    unless the array was regenerated.
    """
    changed = None if previous is None else config_changes(previous, config)
    if changed is not None and pulse_only_change(changed):
        if not changed:
            return "unchanged", None
        full = merged(previous, config)
        await crs.set_pulse_mode(full.get("pulse_mode", "none"),
                                 **pulse_mode_kwargs(full))
        return "pulses", None
    if config.get("resonator_random_seed") is None:
        import random
        config["resonator_random_seed"] = random.randint(0, 2**31 - 1)
    count = await crs.generate_resonators(config)
    return "regenerated", count


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
