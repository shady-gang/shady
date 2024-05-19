%include <stdint.i>
//%apply int { _Bool };

%module shady
%{
#include "shady/ir.h"
%}

%include "shady/ir.h"