_isleaf(::KeyPath, x) = MLDataDevices.isleaf(x)
_isleaf(x) = MLDataDevices.isleaf(x)

"""
    _keypath_to_path(kp::KeyPath) -> String

Convert a `Functors.KeyPath` to a POSIX path suitable for Zarr group/array hierarchies.
Symbols are converted to strings, integers are represented as strings.
"""
function _keypath_to_path(kp::KeyPath)
    keys_tuple = kp.keys
    if isempty(keys_tuple)
        return ""
    end
    return join(string.(keys_tuple), "/")
end

"""
    _serialize_keypath(kp::KeyPath) -> Vector{Any}

Serialize a `Functors.KeyPath` into a JSON-compatible array of tokens.
Symbols are prefixed with ":" (e.g. ":weight"), integers remain numbers.
"""
function _serialize_keypath(kp::KeyPath)
    return Any[k isa Symbol ? ":$(k)" : k for k in kp.keys]
end

"""
    _deserialize_keypath(tokens) -> KeyPath

Reconstruct a `Functors.KeyPath` from serialized JSON tokens.
"""
function _deserialize_keypath(tokens)
    parsed_keys = map(tokens) do t
        if t isa AbstractString && startswith(t, ":")
            Symbol(SubString(t, 2))
        elseif t isa Integer
            Int(t)
        elseif t isa AbstractString && tryparse(Int, t) !== nothing
            parse(Int, t)
        else
            Symbol(t)
        end
    end
    return KeyPath(Tuple(parsed_keys)...)
end
