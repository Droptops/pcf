"""Check experimental controls before interpreting simulator results."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('scaling', SCRIPTS / 'offline_scaling_validation.py')
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


@pytest.mark.parametrize('profile', scaling.PROFILES)
def test_expiry_control_cannot_read_a_prior_entry(profile):
    run = scaling.run_session((profile, 'claims', 'placed', 5, 'expired-every-1'))
    assert all(r['usage']['cache_read_input_tokens'] == 0 for r in run['rows'])
    assert all(r['expired_gap'] for r in run['rows'][1:])


@pytest.mark.parametrize('profile', scaling.PROFILES)
def test_shared_prefix_and_namespace_controls(profile):
    shared = scaling.fleet(profile, 'shared-prefix', sessions=3)['rows']
    assert shared[0]['usage']['cache_read_input_tokens'] == 0
    assert all(r['usage']['cache_read_input_tokens'] > 0 for r in shared[1:])
    for mode in ('unique-system', 'isolated-namespace'):
        isolated = scaling.fleet(profile, mode, sessions=3)['rows']
        assert all(r['usage']['cache_read_input_tokens'] == 0 for r in isolated)
        assert sum(r['input_cost_units'] for r in shared) < sum(r['input_cost_units'] for r in isolated)
