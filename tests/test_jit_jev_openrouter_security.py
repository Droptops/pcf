import pytest

from pcf.jit.jev_openrouter import OpenRouterJevClient


def test_api_key_cannot_be_redirected_to_non_openrouter_host():
    with pytest.raises(ValueError, match="openrouter.ai"):
        OpenRouterJevClient("secret", endpoint="https://example.com/api/alpha/decisions")
    with pytest.raises(ValueError, match="openrouter.ai"):
        OpenRouterJevClient("secret", endpoint="https://openrouter.ai.evil.example/decisions")
