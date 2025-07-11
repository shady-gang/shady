int* provide(void);

int test() {
    int* p = provide();
    return *p;
}