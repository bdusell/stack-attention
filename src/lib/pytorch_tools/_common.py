def apply_to_first_element(func, obj):
    # Allows inputs or outputs with multiple values to be handled gracefully,
    # by assuming that the transformation should only be applied to the first
    # element in case it is a tuple.
    if isinstance(obj, tuple):
        obj, *rest = obj
        return (func(obj), *rest)
    else:
        return func(obj)
