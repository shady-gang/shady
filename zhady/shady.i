%include <stdint.i>
//%apply int { _Bool };

%module shady
%{
#include "shady/ir.h"
#include "shady/runtime.h"
#include "shady/driver.h"
%}

%include "shady/ir.h"
%include "shady/driver.h"
%include "shady/runtime.h"