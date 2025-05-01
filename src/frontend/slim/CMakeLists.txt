shady_add_component(NAME slim_parser SOURCES slim_driver.c parser.c token.c bind.c normalize.c infer.c)
target_link_libraries(slim_parser PUBLIC common api)
target_link_libraries(slim_parser PRIVATE shady)
target_include_directories(slim_parser PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(driver PRIVATE slim_parser)

generate_extinst_headers(SlimFrontendOps extinst.spv-shady-slim-frontend.grammar.json)
target_link_libraries(slim_parser PRIVATE $<BUILD_INTERFACE:SlimFrontendOps>)
