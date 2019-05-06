def all_ancestor(tree_addresses):
    ancestors = set()
    for tree_address in tree_addresses :
        segment = tree_address.split('.')
        base = segment[0]
        ancestors.add(base)
        for i in range(1,len(segment)):
            base = base + '.' + segment[i]
            ancestors.add(base)
    return ancestors


def DA_score(ancestors,tree_addresses):
    DELTA = 0.5
    total_score = 0.0
    for tree_address in tree_addresses :
        self_length = len(tree_address.split('.'))
        for ancestor in ancestors :
            self_segment = tree_address.split('.')
            ance_segment = ancestor.split('.')
            i = 0
            while self_segment[i] == ance_segment[i] :
                i = i + 1
                if (i == len(self_segment))or(i == len(ance_segment)):
                    break
            if i == 0 :
                continue
            if (i == len(self_segment))and(len(ance_segment) > len(self_segment)):
                continue
            da = 1.0
            for j in range(i,self_length):
                da = da * DELTA
            total_score = total_score + da
    return total_score


def semantic_similarity(d1,d2):
    ancestors1 = all_ancestor(d1)
    ancestors2 = all_ancestor(d2)
    union_ancestor = ancestors1 & ancestors2
    da1 = DA_score(ancestors1,d1)
    da2 = DA_score(ancestors2,d2)
    uda1 = DA_score(union_ancestor,d1)
    uda2 = DA_score(union_ancestor,d2)
    similarity = (uda1+uda2)/(da1+da2)
    return similarity

