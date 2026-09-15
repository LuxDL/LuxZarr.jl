_compute_chunks(::Nothing, arr::AbstractArray) = size(arr)
_compute_chunks(f::Function, arr::AbstractArray) = f(arr)
_compute_chunks(chunks::Union{Tuple, AbstractVector}, arr::AbstractArray) = length(chunks) == ndims(arr) ? Tuple(min(c, s) for (c, s) in zip(chunks, size(arr))) : size(arr)
_compute_chunks(chunks, arr::AbstractArray) = size(arr)

function _get_or_create_subgroup(parent_g::Zarr.ZGroup, name::AbstractString)
    sname = String(name)
    if haskey(parent_g.groups, sname)
        return parent_g.groups[sname]
    end
    try
        g = parent_g[sname]
        if g isa Zarr.ZGroup
            return g
        end
    catch
    end
    return Zarr.zgroup(parent_g, sname)
end

_open_root_group(store_or_path::Zarr.ZGroup) = store_or_path
_open_root_group(store_or_path::AbstractString) = Zarr.zopen(String(store_or_path), "r")
_open_root_group(store_or_path) = Zarr.zopen(store_or_path, "r")

_join_zarr_path(prefix::AbstractString, rel_path::AbstractString) = isempty(rel_path) ? String(prefix) : string(prefix, '/', rel_path)

function _get_zarr_node(root_g::Zarr.ZGroup, path::AbstractString)
    isempty(path) && return root_g
    curr = root_g
    for p in eachsplit(path, '/')
        isempty(p) && continue
        if curr isa Zarr.ZGroup
            sname = String(p)
            if haskey(curr.groups, sname)
                curr = curr.groups[sname]
            elseif haskey(curr.arrays, sname)
                curr = curr.arrays[sname]
            else
                try
                    curr = curr[sname]
                catch
                    return nothing
                end
            end
        else
            return nothing
        end
    end
    return curr
end
