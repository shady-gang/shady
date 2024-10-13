#include "shady/ir.h"
#include "shady/driver.h"

#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x, failure_handler) { if (!(x)) { shd_error_print(#x " failed\n"); failure_handler; } }

static bool check_same_bytes(char* a, char* b, size_t size) {
    if (memcmp(a, b, size) == 0)
        return true;
    printf("Error - different bytes obtained !\n");
    size_t byte = 0;
    printf("addr | bytes of a              | bytes of b\n");
    while (byte < size) {
        printf("%04zx |", byte);
        size_t obyte = byte;
        for (int i = 0; i < 8 && byte < size; byte++, i++) {
            printf(" %02x", a[byte]);
        }
        printf(" |");
        byte = obyte;
        for (int i = 0; i < 8 && byte < size; byte++, i++) {
            printf(" %02x", b[byte]);
        }
        printf("\n");
    }
    return false;
}

static void check_int_literal_against_reference(IrArena* a, const Node* lit, IntLiteral reference) {
    const IntLiteral* ptr = shd_resolve_to_int_literal(lit);
    CHECK(ptr, exit(-1));
    IntLiteral got = *ptr;
    CHECK(got.is_signed == reference.is_signed, exit(-1));
    CHECK(got.width == reference.width, exit(-1));
    CHECK(check_same_bytes((char*) &got.value, (char*) &reference.value, sizeof(got.value)), exit(-1));
    const Node* inserted = int_literal(a, reference);
    CHECK(inserted == lit, exit(-1));
}

static void test_int_literals(IrArena* a) {
    IntLiteral ref_zero_u8 = {
        .width = IntTy8,
        .is_signed = false,
        .value = 0
    };
    check_int_literal_against_reference(a, shd_uint8_literal(a, 0), ref_zero_u8);
    IntLiteral ref_one_u8 = {
        .width = IntTy8,
        .is_signed = false,
        .value = 1
    };
    check_int_literal_against_reference(a, shd_uint8_literal(a, 1), ref_one_u8);
    IntLiteral ref_one_i8 = {
        .width = IntTy8,
        .is_signed = true,
        .value = 1
    };
    check_int_literal_against_reference(a, shd_int8_literal(a, 1), ref_one_i8);
    IntLiteral ref_minus_one_i8 = {
        .width = IntTy8,
        .is_signed = true,
        .value = 255
    };
    check_int_literal_against_reference(a, shd_int8_literal(a, -1), ref_minus_one_i8);
    // Check sign extension works right
    int64_t i64_test_values[] = { 0, 1, 255, 256, -1, 65536, 65535, INT64_MAX, INT64_MIN };
    for (size_t i = 0; i < sizeof(i64_test_values) / sizeof(i64_test_values[0]); i++) {
        int64_t test_value = i64_test_values[i];
        IntLiteral reference_literal = {
            .value = test_value,
            .width = IntTy64,
            .is_signed = true
        };
        uint64_t extracted_literal_value = shd_get_int_literal_value(reference_literal, true);
        int16_t reference_minus_one_i16 = test_value;
        CHECK(check_same_bytes((char*) &extracted_literal_value, (char*) &reference_minus_one_i16, sizeof(uint16_t)), exit(-1));
        uint64_t minus_one_u32 = shd_get_int_literal_value(reference_literal, true);
        int32_t reference_minus_one_i32 = test_value;
        CHECK(check_same_bytes((char*) &minus_one_u32, (char*) &reference_minus_one_i32, sizeof(uint32_t)), exit(-1));
        uint64_t minus_one_u64 = shd_get_int_literal_value(reference_literal, true);
        int64_t reference_minus_one_i64 = test_value;
        CHECK(check_same_bytes((char*) &minus_one_u64, (char*) &reference_minus_one_i64, sizeof(uint64_t)), exit(-1));
    }
}

int main(int argc, char** argv) {
    shd_parse_common_args(&argc, argv);

    TargetConfig target_config = shd_default_target_config();
    ArenaConfig aconfig = shd_default_arena_config(&target_config);
    aconfig.check_types = true;
    aconfig.allow_fold = true;
    IrArena* a = shd_new_ir_arena(&aconfig);
    test_int_literals(a);
    shd_destroy_ir_arena(a);
}
