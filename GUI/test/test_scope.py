def test_scope():
    for attempt in range(1):
        try:
            raw = 1 / 0
        except Exception:
            pass
    print("Value of raw:", raw)

test_scope()
