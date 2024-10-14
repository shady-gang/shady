# Code style Guidelines

This guide is for our own reference as much as anyone else.
If you have a good reason for breaking any of those rules we're happy to consider your contributions without prejudice.

 * 4 spaces indentation
 * `Type* name`
 * `snake_case` by default
 * `UpperCamelCase` for types and typedefs
 * Use only typedefs or append `_` to struct/union/enums names
 * `LOUD_CASE` for public macros
 * `{ spaces, surrounding, initializer, lists }`
 * Unless you're literally contributing using a 80-column display (for which I'll ask visual proof), don't format your code as if you do.
 * Include order: 
   * If appropriate, (Private) `self.h` should always come first, with other local headers in the same group
   * Then other `shady/` headers
   * Then in-project utility headers
   * Then external headers
   * Finally, standard C library headers.
   * Each category of includes spaced by an empty line

## Symbol naming

Due to C not having namespaces, we have to deal with this painfully automatable problem ourselves.

 * Avoid exposing any symbols that don't need to be exposed (use `static` wherever you can)
 * Prefixes:
   * `shd_` in front of API functions (in the root `include` folder)
     * `slim_`, `l2s_`, `spv_` and `vcc_` are used in various sub-projects
   * `subsystem_` is acceptable for internal use
   * `shd_subsystem_` is preferable where a clear delineation can be made
   * `shd_new_thing` and `shd_destroy_thing` for constructors and destructors
   * `static inline` functions in headers are not required to be prefixed
   * Types & Typedefs may be prefixed with `Shd`
     * Alternatively, subsystem-relevant ones can use another prefix, much like for functions
* Do not expose global variables to external APIs at all (provide getter functions if necessary)

## Cursing in comments

Can be funny but keep it somewhat family friendly.