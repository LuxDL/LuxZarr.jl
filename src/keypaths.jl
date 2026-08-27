_isleaf(::KeyPath, x) = MLDataDevices.isleaf(x)
_isleaf(x) = MLDataDevices.isleaf(x)

"""
    _keypath_to_path(kp::KeyPath) -> String

Convert a `Functors.KeyPath` to a POSIX path suitable for Zarr group/array hierarchies.
Symbols are converted to strings, integers are represented as strings.
"""
function _keypath_to_path(kp::KeyPath)
    isempty(kp.keys) && return ""
    return join(kp.keys, '/')
end

_serialize_token(k::Symbol) = ":$(k)"
_serialize_token(k::Integer) = Int(k)
_serialize_token(k) = string(k)

"""
    _serialize_keypath(kp::KeyPath) -> Vector{Any}

Serialize a `Functors.KeyPath` into a JSON-compatible array of tokens.
Symbols are prefixed with ":" (e.g. ":weight"), integers remain numbers.
"""
_serialize_keypath(kp::KeyPath) = Any[_serialize_token(k) for k in kp.keys]

_deserialize_token(t::Symbol) = t
_deserialize_token(t::Integer) = Int(t)
function _deserialize_token(t::AbstractString)
    if startswith(t, ':')
        return Symbol(SubString(t, 2))
    end
    val = tryparse(Int, t)
    return val !== nothing ? val : Symbol(t)
end

"""
    _deserialize_keypath(tokens) -> KeyPath

Reconstruct a `Functors.KeyPath` from serialized JSON tokens.
"""
function _deserialize_keypath(tokens)
    return KeyPath(Tuple(map(_deserialize_token, tokens))...)
end
