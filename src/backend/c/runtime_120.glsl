int mod(int a, int b) {
    int q = a / b;
    int qb = q * b;
    return a - qb;
}

int rshift1(int word) {
    return word / 2;
}

int lshift1(int word) {
    int capped = mod(word, 32768);
    return capped * 2;
}

bool extract_bit(int word, int pos) {
    int shifted = word;
    for (int i = 0; i < pos; i++) {
        shifted = rshift1(shifted);
    }
    return mod(shifted, 2) == 1;
}	

const int bits[16] = int[16](0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000);

int set_bit(int word, int pos, bool value) {
    int result = 0;
    for (int i = 0; i < 16; i++) {
        bool set;
        if (i == pos)
            set = value;
        else
            set = extract_bit(word, i);
        if (set)
            result += bits[i];
    }
    return result;
    //if (value) {
    //    return mod(unset + bits[pos], 65536);
    //}
    //return unset;
}

int and(int a, int b) {
    int shifteda = a;
    int shiftedb = b;
    int result = 0;
    for (int i = 0; i < 16; i++) {
        bool ba = mod(shifteda, 2) == 1;
        bool bb = mod(shiftedb, 2) == 1;
        bool br = ba && bb;
        
        if (br)
            result += bits[i];

        shifteda = rshift1(shifteda);
        shiftedb = rshift1(shiftedb);
    }
    return result;
}

int or(int a, int b) {
    int shifteda = a;
    int shiftedb = b;
    int result = 0;
    for (int i = 0; i < 16; i++) {
        bool ba = mod(shifteda, 2) == 1;
        bool bb = mod(shiftedb, 2) == 1;
        bool br = ba || bb;
        
        if (br)
            result += bits[i];

        shifteda = rshift1(shifteda);
        shiftedb = rshift1(shiftedb);
    }
    return result;
}

int xor(int a, int b) {
    int shifteda = a;
    int shiftedb = b;
    int result = 0;
    for (int i = 0; i < 16; i++) {
        bool ba = mod(shifteda, 2) == 1;
        bool bb = mod(shiftedb, 2) == 1;
        bool br = ba ^^ bb;
        
        if (br)
            result += bits[i];

        shifteda = rshift1(shifteda);
        shiftedb = rshift1(shiftedb);
    }
    return result;
}

int not(int a) {
    int shifteda = a;
    int result = 0;
    for (int i = 0; i < 16; i++) {
        bool ba = mod(shifteda, 2) == 1;
        bool br = !ba;
        
        if (br)
            result += bits[i];

        shifteda = rshift1(shifteda);
    }
    return result;
}

bool and(bool a, bool b) { return a && b; }
bool  or(bool a, bool b) { return a || b; }
bool xor(bool a, bool b) { return a ^^ b; }
bool not(bool a) { return !a; }

