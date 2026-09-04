"""A build reports where it is, and can be asked while it runs: its
work is on a thread, not on the server's event loop."""
import asyncio
import contextlib
import io


def test_progress_is_answerable_during_the_build():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    seen = []

    async def poll(build):
        while not build.done():
            seen.append(await crs.get_build_progress())
            await asyncio.sleep(0.005)

    async def main():
        build = asyncio.ensure_future(crs.generate_resonators({
            "num_resonances": 6, "resonator_random_seed": 3,
            "auto_bias_kids": True}))
        await poll(build)
        return await build

    with contextlib.redirect_stdout(io.StringIO()):
        count, freqs = asyncio.run(main())
    assert count == 6
    stages = [p["stage"] for p in seen]
    assert "generating" in stages and "biasing" in stages, stages
    biasing = [p for p in seen if p["stage"] == "biasing"]
    assert all(p["total"] == 6 for p in biasing)
    assert max(p["done"] for p in biasing) >= 1
    final = asyncio.run(crs.get_build_progress())
    assert final == {"stage": "done", "done": 6, "total": 6}


def test_auto_bias_is_not_capped_at_256():
    """Only the packet's channel count bounds how many resonators get a
    tone; the old cap of 256 left the rest of a large array unbiased."""
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    freqs = [1.0e9 + 5e6 * k for k in range(300)]
    calls = []

    async def fake_set_frequency(freq, channel=None, module=None):
        calls.append(channel)

    async def noop(*a, **k):
        return None
    crs.set_frequency = fake_set_frequency
    crs.set_amplitude = noop
    crs.set_phase = noop
    crs.set_nco_frequency = noop
    crs._find_s21_dip_frequency = lambda f, amp: f
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs._auto_bias_kids({"bias_amplitude": 0.001}, freqs))
    assert max(calls) == 300
