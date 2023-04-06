add_executable(aobench_host ao_host.c ao_main.c)
target_link_libraries(aobench_host m shady runtime common)