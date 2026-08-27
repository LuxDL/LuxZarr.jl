function _compute_chunks(chunks, arr::AbstractArray)
    if chunks === nothing
        return size(arr)
    elseif chunks isa Function
        return chunks(arr)
    elseif chunks isa Union{Tuple, AbstractVector} && length(chunks) == ndims(arr)
        return Tuple(min(c, s) for (c, s) in zip(chunks, size(arr)))
    else
        return size(arr)
    end
end

function _get_or_create_subgroup(parent_g::Zarr.ZGroup, name::String)
    try
        g = parent_g[name]
        if g isa Zarr.ZGroup
            return g
        end
    catch
    end
    return Zarr.zgroup(parent_g, name)
end

function _open_root_group(store_or_path)
    if store_or_path isa Zarr.ZGroup
        return store_or_path
    elseif store_or_path isa AbstractString
        return Zarr.zopen(String(store_or_path), "r")
    else
        return Zarr.zopen(store_or_path, "r")
    end
end

function _get_zarr_node(root_g::Zarr.ZGroup, path::String)
    parts = filter(!isempty, split(path, "/"))
    curr = root_g
    for p in parts
        if curr isa Zarr.ZGroup
            try
                curr = curr[p]
            catch
                return nothing
            end
        else
            return nothing
        end
    end
    return curr
end
