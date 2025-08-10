
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )
    return numerator // denominator
