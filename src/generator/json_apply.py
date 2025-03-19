import copy

def json_apply_object(target, src):
    assert (type(target) is dict)
    assert (type(src) is dict)
    for name, value in src.items():
        existing = target.get(name)
        if existing is not None and type(existing) is dict:
            json_apply_object(existing, value)
        elif existing is not None and type(existing) is list and type(value) is list:
            assert existing != value
            for elem in value:
                existing.append(elem)
        else:
            if existing is not None:
                print(f"json-apply: overwriting key '{name}'")
            target[name] = copy.deepcopy(value)