from app.core.security import RateLimiter


def test_requests_under_the_limit_are_allowed():
    limiter = RateLimiter(limit_per_minute=3)

    assert [limiter.check("client") for _ in range(3)] == [True, True, True]


def test_the_request_over_the_limit_is_refused():
    limiter = RateLimiter(limit_per_minute=2)
    limiter.check("client")
    limiter.check("client")

    assert limiter.check("client") is False


def test_clients_are_limited_independently():
    limiter = RateLimiter(limit_per_minute=1)
    limiter.check("first")

    assert limiter.check("second") is True


def test_the_window_slides_so_old_hits_stop_counting(monkeypatch):
    limiter = RateLimiter(limit_per_minute=1)
    clock = [1000.0]
    monkeypatch.setattr("app.core.security.time.monotonic", lambda: clock[0])

    assert limiter.check("client") is True
    assert limiter.check("client") is False

    clock[0] += 61  # past the 60-second window
    assert limiter.check("client") is True


def test_reset_clears_all_recorded_hits():
    limiter = RateLimiter(limit_per_minute=1)
    limiter.check("client")

    limiter.reset()

    assert limiter.check("client") is True
