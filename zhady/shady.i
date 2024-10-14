%include "common.i"
%include <stdint.i>
//%apply int { _Bool };

%module shady
%{
#include "shady/ir.h"

* bb

* bb

#include "shady/runtime.h"

* d

* r

* runtime

#include "shady/driver.h"

* mod

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