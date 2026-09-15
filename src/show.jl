function _format_bytes(bytes::Real)
    bytes < 1024 && return "$(round(Int, bytes)) B"
    bytes < 1024^2 && return "$(round(bytes / 1024, digits = 1)) KB"
    bytes < 1024^3 && return "$(round(bytes / 1024^2, digits = 1)) MB"
    return "$(round(bytes / 1024^3, digits = 2)) GB"
end

_format_count(n::Integer) = reverse(join([join(reverse(chunk)) for chunk in Iterators.partition(reverse(string(n)), 3)], ","))

struct _TreeStats
    total_params::Int
    total_bytes::Int
    num_lazy::Int
    num_in_memory::Int
end

function _collect_tree_stats(tree)
    total_params, total_bytes, num_lazy, num_in_memory = 0, 0, 0, 0
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
        end
    end
    _walk(tree)
    return _TreeStats(total_params, total_bytes, num_lazy, num_in_memory)
end

_to_entries(x::AbstractDict) = collect(pairs(x))
_to_entries(x::NamedTuple) = collect(pairs(x))
_to_entries(x) = collect(enumerate(x))

function _print_tree_node(io::IO, name::String, val, prefix::String, is_last::Bool, is_all_lazy::Bool)
    branch = is_last ? "└── " : "├── "
    next_prefix = prefix * (is_last ? "    " : "│   ")

    return if val isa NamedTuple || val isa AbstractDict || (val isa Tuple && !isempty(val) && first(val) isa Union{NamedTuple, AbstractDict, Tuple})
        println(io, prefix, branch, name)
        entries = _to_entries(val)
        for (i, (k, v)) in enumerate(entries)
            _print_tree_node(io, string(k), v, next_prefix, i == length(entries), is_all_lazy)
        end
    elseif val isa AbstractArray
        tag = is_all_lazy || val isa Zarr.ZArray ? "" : "  [in-memory $(nameof(typeof(val)))]"
        println(io, prefix, branch, name, ": ", eltype(val), " ", size(val), tag)
    elseif !isnothing(val)
        println(io, prefix, branch, name, ": ", sprint(show, val))
    end
end

function _show_tree(io::IO, data; title::String = "", unit::String = "parameters", status_suffix::String = "", stats::_TreeStats = _collect_tree_stats(data))
    is_all_lazy = stats.num_in_memory == 0 && stats.num_lazy > 0

    header = if !isempty(title)
        title
    elseif is_all_lazy
        "LazyParameters (all parameters on-disk):"
    elseif stats.num_lazy > 0 && stats.num_in_memory > 0
        "LazyParameters ($(_format_count(stats.num_lazy)) on-disk / $(_format_count(stats.num_in_memory)) in-memory):"
    else
        "LazyParameters:"
    end
    println(io, header)

    entries = _to_entries(data)
    for (i, (k, v)) in enumerate(entries)
        _print_tree_node(io, string(k), v, "", i == length(entries), is_all_lazy)
    end

    return if stats.total_params > 0
        println(io, "─"^45)
        suffix = isempty(status_suffix) ? "" : " | " * (is_all_lazy ? "All on-disk" : status_suffix)
        print(io, "$(_format_count(stats.total_params)) $unit ($(_format_bytes(stats.total_bytes)))", suffix)
    end
end

Base.show(io::IO, ::MIME"text/plain", lp::LazyParameters) = _show_tree(io, unwrap(lp))
Base.show(io::IO, ::MIME"text/plain", ls::LazyState) = _show_tree(io, unwrap(ls); title = "LazyState:", unit = "state elements")
function Base.show(io::IO, ::MIME"text/plain", lm::LazyLuxModel)
    stats = _collect_tree_stats(unwrap(lm.ps))
    status = "$(_format_count(stats.num_lazy)) on-disk / $(_format_count(stats.num_in_memory)) in-memory"
    return _show_tree(io, unwrap(lm.ps); title = "LazyLuxModel ($(nameof(typeof(lm.model)))):", status_suffix = status, stats = stats)
end

Base.show(io::IO, lp::LazyParameters) = print(io, "LazyParameters(", length(keys(lp)), " entries)")
Base.show(io::IO, ls::LazyState) = print(io, "LazyState(", length(keys(ls)), " entries)")
Base.show(io::IO, lm::LazyLuxModel) = print(io, "LazyLuxModel(", nameof(typeof(lm.model)), ")")
