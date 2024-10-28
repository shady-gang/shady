template<typename T>
T uniform __shady_extract(T varying t, uniform int i) {
    varying T* uniform p1 = &t;
    uniform T* uniform p2 = (uniform T* uniform) p1;
    return p2[i];
}
