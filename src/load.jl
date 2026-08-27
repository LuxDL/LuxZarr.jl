"""
    load_model(store_or_path, [model / ps, st]; lazy=true, metadata_only=false, kwargs...)

Load model parameters, states, and optionally model architecture from a Zarr store or directory.

## Modes

1. **Guided loading into initialized `(ps, st)`**:
   ```julia
   ps, st = load_model(store_or_path, ps, st; lazy=true, kwargs...)
   ```
   Populates existing `(ps, st)` matching KeyPaths.

2. **Standalone loading**:
   ```julia
   # When model architecture is reconstructable:
   lazy_model = load_model(store_or_path) # Returns LazyLuxModel(model, ps, st)
   # Or without model architecture:
   ps, st = load_model(store_or_path)     # Returns (LazyParameters, LazyState)
   ```

3. **Guided loading into a model**:
   ```julia
   lazy_model = load_model(store_or_path, model; lazy=true, kwargs...)
   ```
   (Available when `Lux` is loaded).

## Keyword Arguments

  - `lazy`: If `true` (default), leaves array nodes as lazy `Zarr.ZArray` leaves wrapped in `LazyParameters` and `LazyState` (or `LazyLuxModel`). If `false`, immediately materializes all arrays into memory and returns raw unwrapped `NamedTuple`s.
  - `metadata_only`: If `true`, returns only the metadata dictionary without loading tensor arrays into memory. Defaults to `false`.
"""
function load_model end

_wrap_lazy(x, ::Type{T}, lazy::Bool) where {T} = lazy ? T(x) : x

# Mode 1: Guided load with (ps, st)
function load_model(
    store_or_path,
    ps,
    st=NamedTuple();
    lazy::Bool=true,
    metadata_only::Bool=false,
    kwargs...,
)
    root_g = _open_root_group(store_or_path)
    attrs = Dict{String, Any}(root_g.attrs)
    metadata_only && return attrs

    scalar_params = get(attrs, "scalar_parameters", Dict{String, Any}())
    scalar_states = get(attrs, "scalar_states", Dict{String, Any}())

    ps_loaded = _guided_load_tree(root_g, unwrap(ps), "parameters", scalar_params, lazy)
    st_loaded = _guided_load_tree(root_g, unwrap(st), "states", scalar_states, lazy)

    return (_wrap_lazy(ps_loaded, LazyParameters, lazy), _wrap_lazy(st_loaded, LazyState, lazy))
end

# Mode 2: Standalone load without model or ps/st
function load_model(
    store_or_path;
    lazy::Bool=true,
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

    ps = _reconstruct_from_keypaths(_load_tree_entries(root_g, "parameters", param_keypaths, scalar_params, lazy))
    st = _reconstruct_from_keypaths(_load_tree_entries(root_g, "states", state_keypaths, scalar_states, lazy))
    model = reconstruct_model_from_info(get(attrs, "model_info", nothing))

    if model !== nothing && isdefined(LuxZarr, :_setup_model_skeleton)
        ps, st = _setup_model_skeleton(model, root_g, ps, st, scalar_states, lazy)
        return lazy ? LazyLuxModel(model, LazyParameters(ps), LazyState(st)) : (ps, st, model)
    end

    return (_wrap_lazy(ps, LazyParameters, lazy), _wrap_lazy(st, LazyState, lazy))
end

function _guided_load_tree(root_g::Zarr.ZGroup, tree, prefix::String, scalar_dict::AbstractDict, lazy::Bool)
    return Functors.fmap_with_path(tree; exclude=_isleaf) do kp, x
        rel_path = _keypath_to_path(kp)
        if x isa AbstractArray
            full_path = isempty(rel_path) ? prefix : string(prefix, '/', rel_path)
            z_node = _get_zarr_node(root_g, full_path)
            if z_node isa Zarr.ZArray
                return lazy ? z_node : adapt(typeof(x), convert(AbstractArray{eltype(x)}, Array(z_node)))
            end
        elseif haskey(scalar_dict, rel_path)
            return _deserialize_scalar(scalar_dict[rel_path], typeof(x))
        end
        return x
    end
end

function _load_tree_entries(root_g::Zarr.ZGroup, prefix::String, raw_keypaths, scalar_dict::AbstractDict, lazy::Bool)
    entries = Pair{KeyPath, Any}[]
    for raw_kp in raw_keypaths
        kp = _deserialize_keypath(raw_kp)
        rel_path = _keypath_to_path(kp)
        full_path = isempty(rel_path) ? prefix : string(prefix, '/', rel_path)
        z_node = _get_zarr_node(root_g, full_path)
        if z_node isa Zarr.ZArray
            push!(entries, kp => lazy ? z_node : Array(z_node))
        end
    end
    for (path_str, val) in scalar_dict
        tokens = split(path_str, '/')
        kp = _deserialize_keypath(tokens)
        push!(entries, kp => _deserialize_scalar(val, Any))
    end
    return entries
end
