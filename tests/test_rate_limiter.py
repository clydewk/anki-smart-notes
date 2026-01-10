
import asyncio
import time
import pytest
from src.rate_limiter import MultiDimensionalRateLimiter, ProviderUnavailableError, RateLimitDimension

@pytest.fixture
def mock_limiter():
    limiter = MultiDimensionalRateLimiter(provider="test_provider")
    # Override thresholds for faster testing
    limiter._cb_threshold = 3
    limiter._cb_window = 1.0
    limiter._cb_cooldown = 2.0
    return limiter

@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_threshold(mock_limiter):
    """Circuit trips after consecutive 429s exceed threshold."""
    assert not mock_limiter.is_circuit_open
    
    # 1st failure
    await mock_limiter.report_failure()
    assert not mock_limiter.is_circuit_open
    
    # 2nd failure
    await mock_limiter.report_failure()
    assert not mock_limiter.is_circuit_open
    
    # 3rd failure (threshold reached)
    await mock_limiter.report_failure()
    assert mock_limiter.is_circuit_open

@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success(mock_limiter):
    """Consecutive 429 counter resets after a successful request."""
    # 2 failures
    await mock_limiter.report_failure()
    await mock_limiter.report_failure()
    assert mock_limiter._consecutive_429s == 2
    
    # Success
    await mock_limiter.report_success()
    assert mock_limiter._consecutive_429s == 0
    
    # 3rd failure (should be treated as 1st new failure)
    await mock_limiter.report_failure()
    assert not mock_limiter.is_circuit_open

@pytest.mark.asyncio
async def test_circuit_open_raises_error(mock_limiter):
    """Acquiring when circuit is open raises ProviderUnavailableError."""
    # Trip circuit
    for _ in range(3):
        await mock_limiter.report_failure()
    
    assert mock_limiter.is_circuit_open
    
    with pytest.raises(ProviderUnavailableError):
        await mock_limiter.acquire()

@pytest.mark.asyncio
async def test_circuit_closes_after_cooldown(mock_limiter):
    """Circuit closes after cooldown period."""
    # Trip circuit
    for _ in range(3):
        await mock_limiter.report_failure()
    
    assert mock_limiter.is_circuit_open
    
    # Wait for cooldown
    await asyncio.sleep(2.1)
    
    assert not mock_limiter.is_circuit_open
    # Should be able to acquire again
    await mock_limiter.acquire()

@pytest.mark.asyncio
async def test_daily_quota_trips_immediately(mock_limiter):
    """Daily quota exhaustion trips circuit immediately."""
    assert not mock_limiter.is_circuit_open
    
    # Report daily limit failure
    await mock_limiter.report_failure(is_daily_limit=True)
    
    assert mock_limiter.is_circuit_open
    
@pytest.mark.asyncio
async def test_window_reset(mock_limiter):
    """Counter resets if failures are spaced out."""
    # 1st failure
    await mock_limiter.report_failure()
    assert mock_limiter._consecutive_429s == 1
    
    # Wait for window to pass
    await asyncio.sleep(1.1)
    
    # 2nd failure (should be treated as new sequence)
    await mock_limiter.report_failure()
    assert mock_limiter._consecutive_429s == 1
