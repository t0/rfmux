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
    # Six resonators generate in a few milliseconds, so which stage the
    # poller catches is timing; that it caught the build mid-way at all
    # is the contract, and it did so only because the work is off the
    # event loop.
    stages = [p["stage"] for p in seen]
    mid = [s for s in stages if s not in ("idle", "done")]
    assert mid, stages
    assert set(mid) <= {"generating", "biasing", "warming"}, stages
    final = asyncio.run(crs.get_build_progress())
    assert final == {"stage": "done", "done": 6, "total": 6}


def test_generation_reports_each_resonator():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators({
            "num_resonances": 2, "resonator_random_seed": 3,
            "auto_bias_kids": False}))
    calls = []
    with contextlib.redirect_stdout(io.StringIO()):
        crs._resonator_model.generate_resonators(
            num_resonances=5, config=crs._physics_config,
            progress=lambda done, total: calls.append((done, total)))
    assert calls == [(k, 5) for k in range(5)]


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
