"""The shipped simplified_tuning_flow.py script runs end to end against
the mock."""
from importlib import util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow_acquisition

# parents[2] is the repo root: test/algorithms/<this file>
demo_path = (Path(__file__).resolve().parents[2] / "rfmux"
             / "reference-notebooks" / "Demos" / "simplified_tuning_flow.py")
spec = util.spec_from_file_location("simplified_tuning_flow", demo_path)
simplified_tuning_flow = util.module_from_spec(spec)
spec.loader.exec_module(simplified_tuning_flow)


@pytest.mark.asyncio
async def test_mock_mode_execution():
    assert await simplified_tuning_flow.main(serial="MOCK") == 0
