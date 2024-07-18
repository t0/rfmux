import pytest
import rfmux
import os


@pytest.fixture
def live_session():
    if "CRS_SERIAL" not in os.environ:
        pytest.skip(
            "Set the CRS_SERIAL environment variable to match your CRS board to run this test."
        )

    return rfmux.load_session(
        f"""
        !HardwareMap
        - !CRS {{ serial: "{os.environ['CRS_SERIAL']}" }}
        """
    )
