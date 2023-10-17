#include "l2s_private.h"

#include "dict.h"

ParsedAnnotationContents* find_annotation(Parser* p, const Node* n, AnnotationType t) {
    ParsedAnnotationContents* a = find_value_dict(const Node*, ParsedAnnotationContents, p->annotations, n);
    if (!a)
        return NULL;
    else if (t == NoneAnnot || a->type == t)
        return a;
    else
        return next_annotation(a, t);
}

ParsedAnnotationContents* next_annotation(ParsedAnnotationContents* a, AnnotationType t) {
    do {
        a = a->next;
        if (a && (t == NoneAnnot || a->type == t))
            return a;
    } while (a);
    return a;
}

void add_annotation(Parser* p, const Node* n, ParsedAnnotationContents a) {
    ParsedAnnotationContents* found = find_value_dict(const Node*, ParsedAnnotationContents, p->annotations, n);
    if (found) {
        ParsedAnnotationContents* data = arena_alloc(p->annotations_arena, sizeof(a));
        *data = a;
        while (found->next)
            found = found->next;
        found->next = data;
    } else {
        insert_dict(const Node*, ParsedAnnotationContents, p->annotations, n, a);
    }
}
