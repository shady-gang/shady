#include "emit_spv.h"
#include "emit_spv_builtins.h"

#include "log.h"
#include "portability.h"

VulkanBuiltinKind vulkan_builtins_kind[] = {
#define BUILTIN(kind, name, datatype) VulkanBuiltin##kind,
VULKAN_BUILTINS()
#undef BUILTIN
};

const Type* get_vulkan_builtins_type(IrArena* arena, VulkanBuiltins builtin) {
    switch (builtin) {
#define BUILTIN(kind, name, datatype) case VulkanBuiltin##name: return datatype;
VULKAN_BUILTINS()
#undef BUILTIN
        default: error("Unhandled builtin")
    }
}

// What's the decoration for the builtin
static SpvBuiltIn vulkan_builtins_decoration[] = {
#define BUILTIN(kind, name, datatype) SpvBuiltIn##name,
VULKAN_BUILTINS()
#undef BUILTIN
};

SpvId emit_builtin(Emitter* emitter, VulkanBuiltins builtin) {
    if (emitter->emitted_builtins[builtin] != 0)
        return emitter->emitted_builtins[builtin];

    AddressSpace as = AsInput;
    const Type* builtin_type = get_vulkan_builtins_type(emitter->arena, builtin);
    switch (vulkan_builtins_kind[builtin]) {
        case VulkanBuiltinConstant: error("TODO")
        case VulkanBuiltinOutput: as = AsOutput; SHADY_FALLTHROUGH
        case VulkanBuiltinInput: {
            SpvId id = spvb_fresh_id(emitter->file_builder);
            SpvId type = emit_type(emitter, ptr_type(emitter->arena, (PtrType) { .pointed_type = builtin_type, .address_space = as }));
            spvb_global_variable(emitter->file_builder, id, type, emit_addr_space(as), false, 0);
            uint32_t decoration_payload[] = { vulkan_builtins_decoration[builtin] };
            spvb_decorate(emitter->file_builder, id, SpvDecorationBuiltIn, 1, decoration_payload);
            emitter->emitted_builtins[builtin] = id;
            return id;
        }
        default: error("unreachable")
    }
}
