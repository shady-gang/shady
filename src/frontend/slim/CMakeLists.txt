shady_add_component(NAME frontend_slim SOURCES slim_driver.c parser.c token.c bind.c normalize.c infer.c)
target_link_libraries(frontend_slim PUBLIC common api)
target_link_libraries(frontend_slim PRIVATE shady)
target_include_directories(frontend_slim PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(driver PRIVATE frontend_slim)

generate_extinst_headers(SlimFrontendOps extinst.spv-shady-slim-frontend.grammar.json)
target_link_libraries(frontend_slim PRIVATE $<BUILD_INTERFACE:SlimFrontendOps>)
