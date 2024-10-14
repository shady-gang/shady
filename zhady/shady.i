%include "common.i"
%include <stdint.i>
//%apply int { _Bool };

%module shady
%{
#include "shady/ir.h"

* arena

* arena

* m

* m

* m

* mname

* v

* arena* node

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