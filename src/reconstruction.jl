mutable struct _OrderedNode
    children::Dict{Any, Any}
    keys_order::Vector{Any}
    is_leaf::Bool
    value::Any
end
_OrderedNode() = _OrderedNode(Dict{Any, Any}(), Any[], false, nothing)

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

function _recursively_convert_ordered_node(node::_OrderedNode)
    if node.is_leaf
        return node.value
    end
    if isempty(node.keys_order)
        return NamedTuple()
    end
    if all(k -> k isa Symbol, node.keys_order)
        sym_keys = Tuple(Symbol(k) for k in node.keys_order)
        sym_vals = Tuple(
            node.children[k] isa _OrderedNode ?
            _recursively_convert_ordered_node(node.children[k]) :
            node.children[k]
            for k in sym_keys
        )
        return NamedTuple{sym_keys}(sym_vals)
    elseif all(k -> k isa Integer, node.keys_order)
        sorted_keys = sort(node.keys_order)
        tup_vals = Tuple(
            node.children[k] isa _OrderedNode ?
            _recursively_convert_ordered_node(node.children[k]) :
            node.children[k]
            for k in sorted_keys
        )
        return tup_vals
    else
        sym_keys = Tuple(Symbol(k) for k in node.keys_order)
        sym_vals = Tuple(
            node.children[k] isa _OrderedNode ?
            _recursively_convert_ordered_node(node.children[k]) :
            node.children[k]
            for k in node.keys_order
        )
        return NamedTuple{sym_keys}(sym_vals)
    end
end
