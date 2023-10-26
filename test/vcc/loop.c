int square(int num) {
    for (int i = 0; i < num; i++) {
        if (i == 9)
            break;
        if (i % 2 == 0)
            continue;
        num--;
    }
    return 0;
}
