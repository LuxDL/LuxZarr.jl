struct _OrderedNode
    children::Dict{Any, Any}
    keys_order::Vector{Any}
end
_OrderedNode() = _OrderedNode(Dict{Any, Any}(), Any[])

function _reconstruct_from_keypaths(entries)
    if isempty(entries)
        return NamedTuple()
    end

    root = _OrderedNode()
    for (kp, val) in entries
        keys_tuple = kp.keys
        if isempty(keys_tuple)
            return val
        end
        curr = root
        for k in keys_tuple[1:(end - 1)]
            if !haskey(curr.children, k)
                node = _OrderedNode()
                curr.children[k] = node
                push!(curr.keys_order, k)
            end
            curr = curr.children[k]
        end
        leaf_k = keys_tuple[end]
        if !haskey(curr.children, leaf_k)
            push!(curr.keys_order, leaf_k)
        end
        curr.children[leaf_k] = val
    end

    return _recursively_convert_ordered_node(root)
end

_node_value(node::_OrderedNode) = _recursively_convert_ordered_node(node)
_node_value(val) = val

function _recursively_convert_ordered_node(node::_OrderedNode)
    if isempty(node.keys_order)
        return NamedTuple()
    end
    if all(k -> k isa Integer, node.keys_order)
        sorted_keys = sort(node.keys_order)
        return Tuple(_node_value(node.children[k]) for k in sorted_keys)
    else
        sym_keys = Tuple(Symbol(k) for k in node.keys_order)
        sym_vals = Tuple(_node_value(node.children[k]) for k in node.keys_order)
        return NamedTuple{sym_keys}(sym_vals)
    end
end
