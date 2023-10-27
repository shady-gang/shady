void g(void);

int f(int x) {
    if (x % 2 == 0) {
        g();
    } else {
        g();
    }
    return 0;
}