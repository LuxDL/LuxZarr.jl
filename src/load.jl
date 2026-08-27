"""
    load_model(store_or_path, [model / ps, st]; metadata_only=false, kwargs...)

Load model parameters, states, and optionally model architecture from a Zarr store or directory.

## Modes

1. **Guided loading into initialized `(ps, st)`**:
   ```julia
   ps, st = load_model(store_or_path, ps, st; kwargs...)
   ```
   Populates existing `(ps, st)` matching KeyPaths, adapting to current device and precision.

2. **Standalone loading**:
   ```julia
   ps, st = load_model(store_or_path; kwargs...)
   # or when model is reconstructable:
   ps, st, model = load_model(store_or_path; kwargs...)
   ```
   Directly reconstructs parameters and states from the Zarr hierarchy and metadata attributes.

3. **Guided loading into a model**:
   ```julia
   ps, st = load_model(store_or_path, model; kwargs...)
   ```
   (Available when `Lux` is loaded).

## Keyword Arguments

  - `metadata_only`: If `true`, returns only the metadata dictionary without loading tensor arrays into memory. Defaults to `false`.
"""
function load_model end

# Mode 1: Guided load with (ps, st)
function load_model(
    store_or_path,
    ps,
    st=NamedTuple();
    metadata_only::Bool=false,
    kwargs...,
)
    root_g = _open_root_group(store_or_path)
    attrs = Dict{String, Any}(root_g.attrs)

    if metadata_only
        return attrs
    end

    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    scalar_states = get(attrs, "scalar_states", Dict{String, Any}())

    # Guided parameter loading
    ps_loaded = Functors.fmap_with_path(ps; exclude=_isleaf) do kp, x
        if x isa AbstractArray
            rel_path = _keypath_to_path(kp)
            full_path = "parameters/" * rel_path
            z_node = _get_zarr_node(root_g, full_path)
            if z_node isa Zarr.ZArray
                arr_data = Array(z_node)
                return adapt(typeof(x), convert(AbstractArray{eltype(x)}, arr_data))
            end
        elseif haskey(scalar_params, _keypath_to_path(kp))
            return _deserialize_scalar(scalar_params[_keypath_to_path(kp)], typeof(x))
        end
        return x
    end

    # Guided state loading
    st_loaded = Functors.fmap_with_path(st; exclude=_isleaf) do kp, x
        if x isa AbstractArray
            rel_path = _keypath_to_path(kp)
            full_path = "states/" * rel_path
            z_node = _get_zarr_node(root_g, full_path)
            if z_node isa Zarr.ZArray
                arr_data = Array(z_node)
                return adapt(typeof(x), convert(AbstractArray{eltype(x)}, arr_data))
            end
        elseif haskey(scalar_states, _keypath_to_path(kp))
            return _deserialize_scalar(scalar_states[_keypath_to_path(kp)], typeof(x))
        end
        return x
    end

    return (ps_loaded, st_loaded)
end

# Mode 2: Standalone load without model or ps/st
function load_model(
    store_or_path;
    metadata_only::Bool=false,
    kwargs...,
)
    root_g = _open_root_group(store_or_path)
    attrs = Dict{String, Any}(root_g.attrs)

    if metadata_only
        return attrs
    end

    param_keypaths = get(attrs, "parameter_keypaths", [])
    state_keypaths = get(attrs, "state_keypaths", [])
    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    scalar_states = get(attrs, "scalar_states", Dict{String, Any}())

    # Load parameter arrays
    param_entries = Pair{KeyPath, Any}[]
    for raw_kp in param_keypaths
        kp = _deserialize_keypath(raw_kp)
        rel_path = _keypath_to_path(kp)
        full_path = "parameters/" * rel_path
        z_node = _get_zarr_node(root_g, full_path)
        if z_node isa Zarr.ZArray
            push!(param_entries, kp => Array(z_node))
        end
    end
    for (path_str, val) in scalar_params
        tokens = split(path_str, "/")
        kp = _deserialize_keypath(tokens)
        push!(param_entries, kp => _deserialize_scalar(val, Any))
    end

    # Load state arrays
    state_entries = Pair{KeyPath, Any}[]
    for raw_kp in state_keypaths
        kp = _deserialize_keypath(raw_kp)
        rel_path = _keypath_to_path(kp)
        full_path = "states/" * rel_path
        z_node = _get_zarr_node(root_g, full_path)
        if z_node isa Zarr.ZArray
            push!(state_entries, kp => Array(z_node))
        end
    end
    for (path_str, val) in scalar_states
        tokens = split(path_str, "/")
        kp = _deserialize_keypath(tokens)
        push!(state_entries, kp => _deserialize_scalar(val, Any))
    end

    ps = _reconstruct_from_keypaths(param_entries)
    st = _reconstruct_from_keypaths(state_entries)
    model = reconstruct_model_from_info(get(attrs, "model_info", nothing))

    if model !== nothing && isdefined(LuxZarr, :_setup_model_skeleton)
        ps, st = LuxZarr._setup_model_skeleton(model, root_g, ps, st, scalar_states)
    end

    if model !== nothing
        return (ps, st, model)
    else
        return (ps, st)
    end
end
