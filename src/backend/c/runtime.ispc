template<typename T>
T uniform* uniform extract_ptr(uniform T* varying t, uniform int i) {
    return (uniform T* uniform) extract((intptr_t) t, i);
}
