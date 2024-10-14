%include "common.i"
%include <stdint.i>
//%apply int { _Bool };

%module shady
%{
#include "shady/ir.h"

* bb

* bbyield_typesarg_typesinitial_values

* bbyield_types

* bbyield_types* body

* bbyield_typesinitial_args* body

* bbyield_types* inspecteeliteralscases* default_case

* bbyield_types* condition* true_case* false_case

* bbvalues

* bb* instruction

* bbvalues

* bb* value

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* bb

* a

* a* mem

* a* mem

#include "shady/runtime.h"
#include "shady/driver.h"
#include "shady/config.h"
#include "shady/be/c.h"
#include "shady/be/spirv.h"
#include "shady/be/dump.h"
%}

%include "shady/ir.h"
%include "grammar_generated.h"
%include "shady/driver.h"
%include "shady/runtime.h"
%include "shady/config.h"
%include "shady/be/c.h"
%include "shady/be/spirv.h"
%include "shady/be/dump.h"