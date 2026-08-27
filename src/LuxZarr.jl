module LuxZarr

using Functors: Functors, KeyPath
using LuxCore: LuxCore, AbstractLuxLayer
using MLDataDevices: MLDataDevices, isleaf, cpu_device
using Adapt: adapt
using Zarr: Zarr

# Extension hooks with dispatch
_get_lux_version(::Any) = nothing
function _setup_model_skeleton end

include("keypaths.jl")
include("scalars.jl")
include("reconstruction.jl")
include("zarr_utils.jl")
include("model_metadata.jl")
include("save.jl")
include("load.jl")

export save_model, load_model, extract_model_info, reconstruct_model_from_info

end # module LuxZarr
