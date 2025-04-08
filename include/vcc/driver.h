#ifndef VCC_DRIVER_H
#define VCC_DRIVER_H

#include "shady/driver.h"

typedef struct {
    bool delete_tmp_file;
    bool only_run_clang;
    const char* tmp_filename;
    const char* include_path;
    struct List* clang_options;
} VccConfig;

void vcc_check_clang(void);

VccConfig vcc_init_config(CompilerConfig* compiler_config);
void cli_parse_vcc_args(VccConfig* options, int* pargc, char** argv);
void destroy_vcc_options(VccConfig vcc_options);

void vcc_run_clang(VccConfig* vcc_options, String filename);
Module* vcc_parse_back_into_module(const CompilerConfig*, const TargetConfig*, VccConfig*, String module_name);

#endif
