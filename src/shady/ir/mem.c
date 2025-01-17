#include "shady/ir/grammar.h"
#include "shady/ir/builder.h"

#include <assert.h>

#pragma GCC diagnostic error "-Wswitch"

const Node* shd_get_parent_mem(const Node* mem) {
    assert(is_mem(mem));
    switch (is_mem(mem)) {
        case NotAMem: return NULL;
        case Mem_AbsMem_TAG:
            return NULL;
        case Mem_Call_TAG:
            mem = mem->payload.call.mem;
            return mem;
        case Mem_IndirectCall_TAG:
            mem = mem->payload.indirect_call.mem;
            return mem;
        case Mem_MemAndValue_TAG:
            mem = mem->payload.mem_and_value.mem;
            return mem;
        case Mem_Comment_TAG:
            mem = mem->payload.comment.mem;
            return mem;
        case Mem_StackAlloc_TAG:
            mem = mem->payload.stack_alloc.mem;
            return mem;
        case Mem_LocalAlloc_TAG:
            mem = mem->payload.local_alloc.mem;
            return mem;
        case Mem_Load_TAG:
            mem = mem->payload.load.mem;
            return mem;
        case Mem_Store_TAG:
            mem = mem->payload.store.mem;
            return mem;
        case Mem_CopyBytes_TAG:
            mem = mem->payload.copy_bytes.mem;
            return mem;
        case Mem_FillBytes_TAG:
            mem = mem->payload.fill_bytes.mem;
            return mem;
        case Mem_PushStack_TAG:
            mem = mem->payload.push_stack.mem;
            return mem;
        case Mem_PopStack_TAG:
            mem = mem->payload.pop_stack.mem;
            return mem;
        case Mem_GetStackSize_TAG:
            mem = mem->payload.get_stack_size.mem;
            return mem;
        case Mem_SetStackSize_TAG:
            mem = mem->payload.set_stack_size.mem;
            return mem;
        case Mem_DebugPrintf_TAG:
            mem = mem->payload.debug_printf.mem;
            return mem;
        case Mem_ExtInstr_TAG:
            mem = mem->payload.ext_instr.mem;
            return mem;
    }
}

const Node* shd_get_original_mem(const Node* mem) {
    while (true) {
        const Node* nmem = shd_get_parent_mem(mem);
        if (nmem) {
            mem = nmem;
            continue;
        }
        return mem;
    }
}

const Node* shd_bld_stack_alloc(BodyBuilder* bb, const Type* type) {
    return shd_bld_add_instruction(bb, stack_alloc(shd_get_bb_arena(bb), (StackAlloc) { .type = type, .mem = shd_bld_mem(bb) }));
}

const Node* shd_bld_local_alloc(BodyBuilder* bb, const Type* type) {
    return shd_bld_add_instruction(bb, local_alloc(shd_get_bb_arena(bb), (LocalAlloc) { .type = type, .mem = shd_bld_mem(bb) }));
}

const Node* shd_bld_load(BodyBuilder* bb, const Node* ptr) {
    return shd_bld_add_instruction(bb, load(shd_get_bb_arena(bb), (Load) { .ptr = ptr, .mem = shd_bld_mem(bb) }));
}

void shd_bld_store(BodyBuilder* bb, const Node* ptr, const Node* value) {
    shd_bld_add_instruction(bb, store(shd_get_bb_arena(bb), (Store) { .ptr = ptr, .value = value, .mem = shd_bld_mem(bb) }));
}
