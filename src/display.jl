function _format_bytes(bytes::Real)
    if bytes < 1024
        return "$(round(Int, bytes)) B"
    elseif bytes < 1024^2
        return "$(round(bytes / 1024, digits=1)) KB"
    elseif bytes < 1024^3
        return "$(round(bytes / 1024^2, digits=1)) MB"
    else
        return "$(round(bytes / 1024^3, digits=2)) GB"
    end
end

function _format_count(n::Integer)
    return reverse(join([join(reverse(chunk)) for chunk in Iterators.partition(reverse(string(n)), 3)], ","))
end

struct _TreeStats
    total_params::Int
    total_bytes::Int
    num_lazy::Int
    num_in_memory::Int
    num_scalars::Int
end

function _collect_tree_stats(tree)
    total_params = 0
    total_bytes = 0
    num_lazy = 0
    num_in_memory = 0
    num_scalars = 0

    _walk(x) = begin
        if x isa Zarr.ZArray
            total_params += length(x)
            total_bytes += sizeof(eltype(x)) * length(x)
            num_lazy += 1
        elseif x isa AbstractArray
            total_params += length(x)
            total_bytes += sizeof(eltype(x)) * length(x)
            num_in_memory += 1
        elseif x isa NamedTuple || x isa AbstractDict || x isa Tuple
            for v in (x isa AbstractDict ? values(x) : x)
                _walk(v)
            end
        elseif x !== nothing
            num_scalars += 1
        end
    end
    _walk(tree)
    return _TreeStats(total_params, total_bytes, num_lazy, num_in_memory, num_scalars)
end

function _print_tree_node(io::IO, name::String, val, prefix::String, is_last::Bool, is_all_lazy::Bool)
    branch = is_last ? "└── " : "├── "
    next_prefix = prefix * (is_last ? "    " : "│   ")

    if val isa NamedTuple || val isa AbstractDict || (val isa Tuple && !isempty(val) && first(val) isa Union{NamedTuple, AbstractDict, Tuple})
        println(io, prefix, branch, name)
        entries = val isa AbstractDict ? collect(pairs(val)) : (val isa NamedTuple ? collect(pairs(val)) : collect(enumerate(val)))
        for (i, (k, v)) in enumerate(entries)
            _print_tree_node(io, string(k), v, next_prefix, i == length(entries), is_all_lazy)
        end
    elseif val isa AbstractArray
        type_str = string(eltype(val))
        dims_str = string(size(val))
        tag = if is_all_lazy
            ""
        elseif val isa Zarr.ZArray
            ""
        else
            "  [in-memory $(nameof(typeof(val)))]"
        end
        println(io, prefix, branch, name, ": ", type_str, " ", dims_str, tag)
    elseif val !== nothing
        println(io, prefix, branch, name, ": ", sprint(show, val))
    end
end

function Base.show(io::IO, ::MIME"text/plain", lp::LazyParameters)
    data = getfield(lp, :data)
    stats = _collect_tree_stats(data)
    is_all_lazy = stats.num_in_memory == 0 && stats.num_lazy > 0

    header = if is_all_lazy
        "LazyParameters (all parameters on-disk):"
    elseif stats.num_lazy > 0 && stats.num_in_memory > 0
        "LazyParameters ($(_format_count(stats.num_lazy)) on-disk / $(_format_count(stats.num_in_memory)) in-memory):"
    else
        "LazyParameters:"
    end
    println(io, header)

    entries = data isa AbstractDict ? collect(pairs(data)) : (data isa NamedTuple ? collect(pairs(data)) : collect(enumerate(data)))
    for (i, (k, v)) in enumerate(entries)
        _print_tree_node(io, string(k), v, "", i == length(entries), is_all_lazy)
    end

    if stats.total_params > 0
        println(io, "─"^45)
        print(io, "$(_format_count(stats.total_params)) parameters ($(_format_bytes(stats.total_bytes)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ls::LazyState)
    data = getfield(ls, :data)
    stats = _collect_tree_stats(data)
    is_all_lazy = stats.num_in_memory == 0 && stats.num_lazy > 0

    header = "LazyState:"
    println(io, header)

    entries = data isa AbstractDict ? collect(pairs(data)) : (data isa NamedTuple ? collect(pairs(data)) : collect(enumerate(data)))
    for (i, (k, v)) in enumerate(entries)
        _print_tree_node(io, string(k), v, "", i == length(entries), is_all_lazy)
    end

    if stats.total_params > 0
        println(io, "─"^45)
        print(io, "$(_format_count(stats.total_params)) state elements ($(_format_bytes(stats.total_bytes)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", lm::LazyLuxModel)
    model = getfield(lm, :model)
    ps = getfield(lm, :ps)
    stats = _collect_tree_stats(unwrap(ps))
    is_all_lazy = stats.num_in_memory == 0 && stats.num_lazy > 0

    m_name = string(nameof(typeof(model)))
    println(io, "LazyLuxModel ($m_name):")

    ps_data = unwrap(ps)
    entries = ps_data isa AbstractDict ? collect(pairs(ps_data)) : (ps_data isa NamedTuple ? collect(pairs(ps_data)) : collect(enumerate(ps_data)))
    for (i, (k, v)) in enumerate(entries)
        _print_tree_node(io, string(k), v, "", i == length(entries), is_all_lazy)
    end

    if stats.total_params > 0
        println(io, "─"^45)
        status_str = is_all_lazy ? "All on-disk" : "$(_format_count(stats.num_lazy)) on-disk / $(_format_count(stats.num_in_memory)) in-memory"
        print(io, "$(_format_count(stats.total_params)) parameters ($(_format_bytes(stats.total_bytes))) | $status_str")
    end
end
