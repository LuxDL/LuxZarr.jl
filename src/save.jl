"""
    save_model(
        store_or_path,
        ps,
        st=NamedTuple();
        model=nothing,
        metadata=Dict{String, Any}(),
        zarr_format::Int=3,
        force::Bool=false,
        compressor=nothing,
        chunks=nothing,
        kwargs...
    ) -> Any

Save model parameters `ps`, states `st`, and optional `model` architecture metadata to a Zarr store or directory.
This function uses the Zarr v3 specification by default and serializes tensor `KeyPath`s via `Functors.fmap_with_path`,
ensuring cross-version compatibility.

## Arguments

  - `store_or_path`: Directory path (`AbstractString`), or an existing `Zarr.ZGroup` / `Zarr.AbstractStore`.
  - `ps`: Model parameters (e.g. `NamedTuple` returned from `Lux.setup` or any nested parameter structure).
  - `st`: Model states (optional, defaults to `NamedTuple()`).

## Keyword Arguments

  - `model`: Optional model architecture struct. If provided, model architecture information will be saved to the Zarr metadata attributes.
  - `metadata`: Optional `AbstractDict` of user-defined metadata to store in the root Zarr attributes.
  - `zarr_format`: Zarr format specification version (defaults to `3`).
  - `force`: If `true`, overwrites existing files if `store_or_path` is a non-empty directory. Defaults to `false`.
  - `compressor`: Compressor to use for Zarr arrays (e.g. Blosc, Zlib). Defaults to `nothing`.
  - `chunks`: Chunk dimensions tuple or function. Defaults to `nothing` (full array chunking).

## Returns

  - The input `store_or_path`.
"""
function save_model(
        store_or_path,
        ps,
        st = NamedTuple();
        model = nothing,
        metadata = Dict{String, Any}(),
        zarr_format::Int = 3,
        force::Bool = false,
        compressor = nothing,
        chunks = nothing,
        kwargs...,
    )
    meta_dict = if metadata isa AbstractDict
        Dict{String, Any}(string(k) => v for (k, v) in pairs(metadata))
    else
        Dict{String, Any}()
    end

    # 1. Setup store
    store = _setup_store(store_or_path; force = force)

    # 2. Collect parameter and state entries
    param_array_entries, scalar_params = _collect_arrays_and_scalars(ps)
    state_array_entries, scalar_states = _collect_arrays_and_scalars(st)

    # 3. Construct root attributes
    root_attrs = Dict{String, Any}(
        "format" => "lux_zarr",
        "zarr_format" => zarr_format,
        "luxzarr_version" => string(pkgversion(LuxZarr)),
        "julia_version" => string(VERSION),
        "created_at" => time(),
        "parameter_keypaths" => [_serialize_keypath(kp) for (kp, _) in param_array_entries],
        "state_keypaths" => [_serialize_keypath(kp) for (kp, _) in state_array_entries],
        "scalar_parameters" => scalar_params,
        "scalar_states" => scalar_states,
        "user_metadata" => meta_dict,
    )
    lux_v = _get_lux_version(Val(:Lux))
    if isnothing(lux_v) && !isnothing(model)
        lux_v = _get_lux_version(model)
    end
    if !isnothing(lux_v)
        root_attrs["lux_version"] = lux_v
    end

    if !isnothing(model)
        root_attrs["model_info"] = extract_model_info(model)
        root_attrs["model_summary"] = _get_model_summary(model)
    end

    # 4. Create root group
    root_g = _create_root_group(store, zarr_format, root_attrs)

    # Update attrs if group already existed
    for (k, v) in root_attrs
        root_g.attrs[k] = v
    end

    # 5. Write parameter and state arrays
    _write_array_entries!(root_g, "parameters", param_array_entries, chunks, compressor)
    _write_array_entries!(root_g, "states", state_array_entries, chunks, compressor)

    return store_or_path
end

function _setup_store(path::AbstractString; force::Bool)
    p = String(path)
    if isdir(p) && !isempty(readdir(p))
        if force
            rm(p; recursive = true, force = true)
        else
            throw(ArgumentError("Directory $(p) already exists and is not empty. Pass `force=true` to overwrite."))
        end
    end
    mkpath(p)
    return Zarr.DirectoryStore(p)
end
_setup_store(store; force::Bool) = store

_unwrap_tree_containers(x) = x
_unwrap_tree_containers(x::NamedTuple) = map(_unwrap_tree_containers, x)
_unwrap_tree_containers(x::Tuple) = map(_unwrap_tree_containers, x)
function _unwrap_tree_containers(x::AbstractArray)
    if Symbol(nameof(typeof(x))) === :ComponentArray
        if isempty(x) && isempty(propertynames(x))
            return NamedTuple()
        end
        return _unwrap_tree_containers(NamedTuple(x))
    end
    return x
end

function _collect_arrays_and_scalars(tree)
    array_entries = Tuple{KeyPath, Array}[]
    scalar_dict = Dict{String, Any}()
    cpu_dev = cpu_device()
    unwrapped_tree = _unwrap_tree_containers(tree)
    Functors.fmap_with_path(unwrapped_tree; exclude = _isleaf) do kp, x
        if x isa AbstractArray
            if !isempty(x)
                arr = Array(cpu_dev(x))
                push!(array_entries, (kp, arr))
            end
        elseif !isnothing(x)
            scalar_dict[_keypath_to_path(kp)] = _serialize_scalar(x)
        end
        return x
    end
    return array_entries, scalar_dict
end

function _create_root_group(store::Zarr.ZGroup, zarr_format::Int, root_attrs::Dict{String, Any})
    return store
end
function _create_root_group(store, zarr_format::Int, root_attrs::Dict{String, Any})
    if isdefined(Zarr, :ZarrFormat)
        try
            return Zarr.zgroup(store, "", Zarr.ZarrFormat(zarr_format); attrs = root_attrs)
        catch
            return Zarr.zgroup(store, Zarr.ZarrFormat(zarr_format); attrs = root_attrs)
        end
    else
        return Zarr.zgroup(store; attrs = root_attrs)
    end
end

function _write_array_entries!(parent_g::Zarr.ZGroup, subgroup_name::String, entries::Vector{Tuple{KeyPath, Array}}, chunks, compressor)
    isempty(entries) && return nothing
    sub_g = _get_or_create_subgroup(parent_g, subgroup_name)
    for (kp, arr) in entries
        curr_g = sub_g
        keys_tuple = kp.keys
        for k in keys_tuple[1:(end - 1)]
            curr_g = _get_or_create_subgroup(curr_g, string(k))
        end
        leaf_name = string(keys_tuple[end])
        arr_chunks = _compute_chunks(chunks, arr)
        kw = isnothing(compressor) ? (; chunks = arr_chunks) : (; chunks = arr_chunks, compressor = compressor)
        z_arr = Zarr.zcreate(eltype(arr), curr_g, leaf_name, size(arr)...; kw...)
        z_arr[ntuple(_ -> Colon(), ndims(arr))...] = arr
    end
    return nothing
end
