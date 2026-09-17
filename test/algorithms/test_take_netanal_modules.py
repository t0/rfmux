"""take_netanal: its argument checks, and a module list giving concurrent
sweeps with one result per module in order."""

import asyncio
import contextlib
import io

import pytest

from rfmux.core.schema import CRS


@pytest.mark.parametrize("bad", [
    dict(fmin=1100e6, fmax=1000e6),
    dict(fmin=1000e6, fmax=1000e6),
    dict(npoints=1),
    dict(nsamps=0),
    dict(amp=0.0),
    dict(max_chans=0),
    dict(max_span=0.0),
])
def test_rejects_arguments_before_touching_the_board(bad):
    crs = CRS(serial="0000")  # unresolved: any board call would fail differently
    kwargs = dict(amp=0.001, fmin=1000e6, fmax=1100e6, nsamps=2, npoints=100, module=1)
    kwargs.update(bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        asyncio.run(crs.take_netanal(**kwargs))


@pytest.fixture
def mock_crs():
    from rfmux import load_session
    from rfmux.core.schema import CRS

    with contextlib.redirect_stdout(io.StringIO()):
        session = load_session('!HardwareMap\n- !flavour "rfmux.mock"\n'
                               '- !CRS { serial: "0000", hostname: "127.0.0.1" }\n')
        crs = session.query(CRS).one()
        asyncio.run(crs.resolve())
    return crs


def test_module_list_returns_one_result_per_module(mock_crs):
    seen = set()
    results = asyncio.run(mock_crs.take_netanal(
        amp=0.001, fmin=1000e6, fmax=1020e6, nsamps=1, npoints=20, max_chans=32,
        module=[1, 2], progress_callback=lambda module, fraction: seen.add(module)))
    assert isinstance(results, list) and len(results) == 2
    assert all(r['frequencies'].size == 20 for r in results)
    assert seen == {1, 2}
