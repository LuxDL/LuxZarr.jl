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
    )

Save model parameters `ps`, states `st`, and optional `model` architecture metadata to a Zarr store or directory.
This function uses the Zarr v3 specification by default and serializes tensor `KeyPath`s via `Functors.fmap_with_path`,
ensuring cross-version compatibility.

## Arguments

  - `store_or_path`: Directory path (`String`), or an existing `Zarr.ZGroup` / `Zarr.AbstractStore`.
  - `ps`: Model parameters (e.g. `NamedTuple` returned from `Lux.setup` or any nested structure).
  - `st`: Model states (optional, defaults to `NamedTuple()`).

## Keyword Arguments

  - `model`: Optional model architecture struct. If provided, model architecture information will be saved to the Zarr metadata attributes.
  - `metadata`: Optional `Dict{String, Any}` of user-defined metadata to store in the root Zarr attributes.
  - `zarr_format`: Zarr format specification version (defaults to `3`).
  - `force`: If `true`, overwrites existing files if `store_or_path` is a non-empty directory. Defaults to `false`.
  - `compressor`: Compressor to use for Zarr arrays (e.g. Blosc, Zlib).
  - `chunks`: Chunk dimensions tuple or function.
"""
function save_model(
    store_or_path,
    ps,
    st=NamedTuple();
    model=nothing,
    metadata=Dict{String, Any}(),
    zarr_format::Int=3,
    force::Bool=false,
    compressor=nothing,
    chunks=nothing,
    kwargs...,
)
    meta_dict = if metadata isa AbstractDict
        Dict{String, Any}(string(k) => v for (k, v) in pairs(metadata))
    else
        Dict{String, Any}()
    end

    # 1. Setup store
    if store_or_path isa AbstractString
        path = String(store_or_path)
        if isdir(path) && !isempty(readdir(path))
            if force
                rm(path; recursive=true, force=true)
            else
                throw(ArgumentError("Directory $(path) already exists and is not empty. Pass `force=true` to overwrite."))
            end
        end
        mkpath(path)
        store = Zarr.DirectoryStore(path)
    else
        store = store_or_path
    end

    # 2. Collect parameter arrays and scalar parameters
    param_array_entries = Tuple{KeyPath, Array}[]
    scalar_params = Dict{String, Any}()

    Functors.fmap_with_path(ps; exclude=_isleaf) do kp, x
        if x isa AbstractArray
            arr = Array(cpu_device()(x))
            push!(param_array_entries, (kp, arr))
        elseif x !== nothing
            scalar_params[_keypath_to_path(kp)] = _serialize_scalar(x)
        end
        return x
    end

    # 3. Collect state arrays and scalar states
    state_array_entries = Tuple{KeyPath, Array}[]
    scalar_states = Dict{String, Any}()

    Functors.fmap_with_path(st; exclude=_isleaf) do kp, x
        if x isa AbstractArray
            arr = Array(cpu_device()(x))
            push!(state_array_entries, (kp, arr))
        elseif x !== nothing
            scalar_states[_keypath_to_path(kp)] = _serialize_scalar(x)
        end
        return x
    end

    # 4. Construct root attributes
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
    if lux_v === nothing && model !== nothing
        lux_v = _get_lux_version(model)
    end
    if lux_v !== nothing
        root_attrs["lux_version"] = lux_v
    end

    if model !== nothing
        root_attrs["model_info"] = extract_model_info(model)
        root_attrs["model_summary"] = _get_model_summary(model)
    end

    # 5. Create root group
    root_g = if store isa Zarr.ZGroup
        store
    elseif isdefined(Zarr, :ZarrFormat)
        try
            Zarr.zgroup(store, "", Zarr.ZarrFormat(zarr_format); attrs=root_attrs)
        catch
            Zarr.zgroup(store, Zarr.ZarrFormat(zarr_format); attrs=root_attrs)
        end
    else
        Zarr.zgroup(store; attrs=root_attrs)
    end
    # Update attrs if group already existed
    for (k, v) in root_attrs
        root_g.attrs[k] = v
    end

    # 6. Write parameter arrays
    if !isempty(param_array_entries)
        ps_g = _get_or_create_subgroup(root_g, "parameters")
        for (kp, arr) in param_array_entries
            tokens = [string(k) for k in kp.keys]
            curr_g = ps_g
            for t in tokens[1:(end - 1)]
                curr_g = _get_or_create_subgroup(curr_g, t)
            end
            leaf_name = tokens[end]
            arr_chunks = _compute_chunks(chunks, arr)
            kw = Dict{Symbol, Any}(:chunks => arr_chunks)
            if compressor !== nothing
                kw[:compressor] = compressor
            end
            z_arr = Zarr.zcreate(eltype(arr), curr_g, leaf_name, size(arr)...; kw...)
            z_arr[ntuple(_ -> Colon(), ndims(arr))...] = arr
        end
    end

    # 7. Write state arrays
    if !isempty(state_array_entries)
        st_g = _get_or_create_subgroup(root_g, "states")
        for (kp, arr) in state_array_entries
            tokens = [string(k) for k in kp.keys]
            curr_g = st_g
            for t in tokens[1:(end - 1)]
                curr_g = _get_or_create_subgroup(curr_g, t)
            end
            leaf_name = tokens[end]
            arr_chunks = _compute_chunks(chunks, arr)
            kw = Dict{Symbol, Any}(:chunks => arr_chunks)
            if compressor !== nothing
                kw[:compressor] = compressor
            end
            z_arr = Zarr.zcreate(eltype(arr), curr_g, leaf_name, size(arr)...; kw...)
            z_arr[ntuple(_ -> Colon(), ndims(arr))...] = arr
        end
    end

    return store_or_path
end
