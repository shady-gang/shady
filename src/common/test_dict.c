#include "dict.h"
#include "log.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// purposefully bad hash to make sure the collision handling is solid
KeyHash bad_hash_i32(int* i) {
    return *i;
}

bool compare_i32(int* pa, int* pb) {
    return *pa == *pb;
}

#define TEST_ENTRIES 10000

void shuffle(int arr[]) {
    for (int i = 0; i < TEST_ENTRIES; i++) {
        int a = rand() % TEST_ENTRIES;
        int b = rand() % TEST_ENTRIES;
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }
}

int main(int argc, char** argv) {
    srand((int) shd_get_time_nano());
    struct Dict* d = shd_new_set(int, (HashFn) bad_hash_i32, (CmpFn) compare_i32);

    int arr[TEST_ENTRIES];
    for (int i = 0; i < TEST_ENTRIES; i++) {
        arr[i] = i;
    }

    shuffle(arr);

    bool contained[TEST_ENTRIES];
    memset(contained, 0, sizeof(contained));

    for (int i = 0; i < TEST_ENTRIES; i++) {
        bool unique = shd_set_insert_get_result(int, d, arr[i]);
        if (!unique) {
            error("Entry %d was thought to be already in the dict", arr[i]);
        }
        contained[arr[i]] = true;
    }

    shuffle(arr);
    for (int i = 0; i < TEST_ENTRIES; i++) {
        assert(contained[arr[i]]);
        assert(shd_dict_find_key(int, d, arr[i]));
    }

    shuffle(arr);
    for (int i = 0; i < rand() % TEST_ENTRIES; i++) {
        assert(contained[arr[i]]);
        bool removed = shd_dict_remove(int, d, arr[i]);
        assert(removed);
        contained[arr[i]] = false;
    }

    shuffle(arr);
    for (int i = 0; i < TEST_ENTRIES; i++) {
        assert(!!shd_dict_find_key(int, d, arr[i]) == contained[arr[i]]);
    }

    shuffle(arr);
    for (int i = 0; i < TEST_ENTRIES; i++) {
        assert(!!shd_dict_find_key(int, d, arr[i]) == contained[arr[i]]);
        if (!contained[arr[i]]) {
            bool unique = shd_set_insert_get_result(int, d, arr[i]);
            if (!unique) {
                error("Entry %d was thought to be already in the dict", arr[i]);
            }
            contained[arr[i]] = true;
        }
        assert(contained[arr[i]]);
    }

    shuffle(arr);
    for (int i = 0; i < TEST_ENTRIES; i++) {
        assert(contained[arr[i]]);
        assert(shd_dict_find_key(int, d, arr[i]));
    }

    shd_destroy_dict(d);
    return 0;
}
