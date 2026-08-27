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
    metadata_only && return attrs

    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    scalar_states = get(attrs, "scalar_states", Dict{String, Any}())

    ps_loaded = _guided_load_tree(root_g, ps, "parameters", scalar_params)
    st_loaded = _guided_load_tree(root_g, st, "states", scalar_states)

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
    metadata_only && return attrs

    param_keypaths = get(attrs, "parameter_keypaths", [])
    state_keypaths = get(attrs, "state_keypaths", [])
    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    scalar_states = get(attrs, "scalar_states", Dict{String, Any}())

    ps = _reconstruct_from_keypaths(_load_tree_entries(root_g, "parameters", param_keypaths, scalar_params))
    st = _reconstruct_from_keypaths(_load_tree_entries(root_g, "states", state_keypaths, scalar_states))
    model = reconstruct_model_from_info(get(attrs, "model_info", nothing))

    if model !== nothing && isdefined(LuxZarr, :_setup_model_skeleton)
        ps, st = LuxZarr._setup_model_skeleton(model, root_g, ps, st, scalar_states)
    end

    return model !== nothing ? (ps, st, model) : (ps, st)
end

function _guided_load_tree(root_g::Zarr.ZGroup, tree, prefix::String, scalar_dict::AbstractDict)
    return Functors.fmap_with_path(tree; exclude=_isleaf) do kp, x
        rel_path = _keypath_to_path(kp)
        if x isa AbstractArray
            full_path = isempty(rel_path) ? prefix : string(prefix, '/', rel_path)
            z_node = _get_zarr_node(root_g, full_path)
            if z_node isa Zarr.ZArray
                arr_data = Array(z_node)
                return adapt(typeof(x), convert(AbstractArray{eltype(x)}, arr_data))
            end
        elseif haskey(scalar_dict, rel_path)
            return _deserialize_scalar(scalar_dict[rel_path], typeof(x))
        end
        return x
    end
end

function _load_tree_entries(root_g::Zarr.ZGroup, prefix::String, raw_keypaths, scalar_dict::AbstractDict)
    entries = Pair{KeyPath, Any}[]
    for raw_kp in raw_keypaths
        kp = _deserialize_keypath(raw_kp)
        rel_path = _keypath_to_path(kp)
        full_path = isempty(rel_path) ? prefix : string(prefix, '/', rel_path)
        z_node = _get_zarr_node(root_g, full_path)
        if z_node isa Zarr.ZArray
            push!(entries, kp => Array(z_node))
        end
    end
    for (path_str, val) in scalar_dict
        tokens = split(path_str, '/')
        kp = _deserialize_keypath(tokens)
        push!(entries, kp => _deserialize_scalar(val, Any))
    end
    return entries
end
