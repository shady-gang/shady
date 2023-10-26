int f(int x) {
    if (x % 2 == 0) goto even;
    goto odd;

    {
        even:
            goto end;
        odd:
            goto end;
    }
    end:
    return 0;
}