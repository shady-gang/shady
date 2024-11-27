template<typename T>
noinline T uniform __shady_extract(T varying t, uniform int i) {
    assert(i >= 0);
    return extract(t, i);
}

uniform uint64 __shady_lanemask() {
    return lanemask();
}
