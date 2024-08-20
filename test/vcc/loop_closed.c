int square(int num) {
    for (int i = 0; i < num; i++) {
        if (i == 9)
            return i;
        num--;
    }
    return 0;
}
