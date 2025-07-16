float f(void) {
    float x;
    *(int*) &x = 3;
    return x;
}

float g(void) {
    int x;
    x = 3;
    return *(float*)&x;
}
