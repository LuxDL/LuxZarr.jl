using LuxZarr
using Lux
using LuxCore
using Test
using StableRNGs
using Random
using Zarr

@testset "LuxZarr.jl Test Suite" begin
    rng = StableRNG(12345)

    @testset "Pure Parameter Tree Serialization (Without Model)" begin
        ps = (
            layer_1 = (weight = randn(rng, Float32, 4, 2), bias = randn(rng, Float32, 4)),
            layer_2 = (weight = randn(rng, Float32, 1, 4), bias = randn(rng, Float32, 1)),
        )
        st = (
            step = 100,
            active = Val(true),
            mode = :eval,
        )

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "pure_tree.zarr")
            save_model(save_path, ps, st; metadata = Dict("custom_tag" => "pure_tree"))

            # Standalone lazy loading (default)
            ps_loaded, st_loaded = load_model(save_path)
            @test ps_loaded isa LazyParameters
            @test st_loaded isa LazyState
            @test ps_loaded.layer_1.weight == ps.layer_1.weight
            @test ps_loaded.layer_1.bias == ps.layer_1.bias
            @test ps_loaded.layer_2.weight == ps.layer_2.weight
            @test ps_loaded.layer_2.bias == ps.layer_2.bias
            @test st_loaded.step == 100
            @test st_loaded.active == Val(true)
            @test st_loaded.mode == :eval

            # Eager loading (lazy=false)
            ps_eager, st_eager = load_model(save_path; lazy = false)
            @test ps_eager isa NamedTuple
            @test !(ps_eager isa LazyParameters)
            @test ps_eager.layer_1.weight == ps.layer_1.weight

            # Guided loading with ps, st
            ps_guided, st_guided = load_model(save_path, ps, st; lazy = false)
            @test ps_guided.layer_1.weight == ps.layer_1.weight
            @test st_guided.step == 100
        end
    end

    @testset "Dense and Chain Models with Lazy Execution & Display" begin
        model = Chain(
            Dense(4 => 8, relu),
            Dense(8 => 2)
        )
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 3)
        y, _ = model(x, ps, st)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "dense_chain.zarr")
            save_model(save_path, ps, st; model = model, metadata = Dict("tag" => "test_dense"))
            @test isdir(save_path)

            # Standalone load returning LazyLuxModel
            lazy_model = load_model(save_path)
            @test lazy_model isa LazyLuxModel
            @test lazy_model.ps isa LazyParameters
            @test lazy_model.st isa LazyState

            # Callable LazyLuxModel execution
            y_lazy, _ = lazy_model(x)
            @test y ≈ y_lazy

            # Direct model(x, ps, st) execution with LazyParameters
            y_direct, _ = model(x, lazy_model.ps, lazy_model.st)
            @test y ≈ y_direct

            # Device transfer (dev(ps)) and materialize(ps)
            cpu_dev = cpu_device()
            ps_cpu = cpu_dev(lazy_model.ps)
            @test ps_cpu.layer_1.weight isa Matrix{Float32}
            @test ps_cpu.layer_1.weight == ps.layer_1.weight

            # Base.materialize
            ps_mat = materialize(lazy_model.ps)
            @test ps_mat.layer_1.weight isa Matrix{Float32}
            @test ps_mat.layer_1.weight == ps.layer_1.weight

            # Single leaf materialize
            w_mat = materialize(lazy_model.ps.layer_1.weight)
            @test w_mat isa Matrix{Float32}
            @test w_mat == ps.layer_1.weight

            # Positional device materialize
            ps_dev = materialize(cpu_dev, lazy_model.ps)
            @test ps_dev.layer_1.weight isa Matrix{Float32}

            # Base.show display testing (100% Lazy on-disk)
            show_str_ps = sprint(show, MIME("text/plain"), lazy_model.ps)
            @test occursin("LazyParameters (all parameters on-disk):", show_str_ps)
            @test occursin("layer_1", show_str_ps)
            @test occursin("weight: Float32 (8, 4)", show_str_ps)
            @test !occursin("[in-memory", show_str_ps) # No redundant tags when 100% lazy

            show_str_model = sprint(show, MIME("text/plain"), lazy_model)
            @test occursin("LazyLuxModel (Chain):", show_str_model)
            @test occursin("All on-disk", show_str_model)

            # Compact 1-line show
            @test sprint(show, lazy_model.ps) == "LazyParameters(2 entries)"
            @test sprint(show, lazy_model.st) == "LazyState(2 entries)"
            @test sprint(show, lazy_model) == "LazyLuxModel(Chain)"

            # Mixed state display (modify one layer in-memory)
            ps_mixed = LazyParameters(
                (
                    layer_1 = (weight = lazy_model.ps.layer_1.weight, bias = lazy_model.ps.layer_1.bias),
                    layer_2 = (weight = Array(lazy_model.ps.layer_2.weight), bias = Array(lazy_model.ps.layer_2.bias)),
                )
            )
            show_str_mixed = sprint(show, MIME("text/plain"), ps_mixed)
            @test occursin("on-disk /", show_str_mixed)
            @test occursin("[in-memory Array]", show_str_mixed) # Highlights only exceptions

            # Eager load (lazy=false)
            ps_eager, st_eager, model_eager = load_model(save_path; lazy = false)
            @test ps_eager isa NamedTuple
            @test !(ps_eager isa LazyParameters)
            y_eager, _ = model_eager(x, ps_eager, st_eager)
            @test y ≈ y_eager
        end
    end

    @testset "Stateful Layers (BatchNorm)" begin
        model = Chain(
            Dense(4 => 6),
            BatchNorm(6),
            Dense(6 => 2)
        )
        ps, st = Lux.setup(rng, model)
        st = Lux.trainmode(st)

        # Forward pass in trainmode to update running statistics
        x = randn(rng, Float32, 4, 10)
        _, st = model(x, ps, st)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "batchnorm_model.zarr")
            save_model(save_path, ps, st; model = model)

            # Lazy load with model
            lazy_model = load_model(save_path, model)
            @test lazy_model isa LazyLuxModel
            @test lazy_model.st.layer_2.running_mean ≈ st.layer_2.running_mean
            @test lazy_model.st.layer_2.running_var ≈ st.layer_2.running_var
            @test lazy_model.st.layer_2.training == st.layer_2.training

            # State display
            show_str_st = sprint(show, MIME("text/plain"), lazy_model.st)
            @test occursin("LazyState:", show_str_st)
            @test occursin("running_mean", show_str_st)

            # Standalone load
            lazy_standalone = load_model(save_path)
            @test lazy_standalone.ps.layer_1.weight ≈ ps.layer_1.weight
            @test lazy_standalone.st.layer_2.running_mean ≈ st.layer_2.running_mean
        end
    end

    @testset "Complex Nested Containers (Parallel, BranchLayer, SkipConnection)" begin
        model_valid = Chain(
            Dense(4 => 4),
            Parallel(+, Dense(4 => 4), Dense(4 => 4)),
            SkipConnection(Dense(4 => 4), +)
        )
        ps, st = Lux.setup(rng, model_valid)
        x = randn(rng, Float32, 4, 2)
        y, _ = model_valid(x, ps, st)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "complex_model.zarr")
            save_model(save_path, ps, st; model = model_valid)

            # Lazy load
            lazy_model = load_model(save_path, model_valid)
            y_lazy, _ = lazy_model(x)
            @test y ≈ y_lazy
        end
    end

    @testset "Overwrite / force handling" begin
        model = Dense(2 => 2)
        ps, st = Lux.setup(rng, model)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "test_force.zarr")
            save_model(save_path, ps, st)

            # Without force=true on existing directory should error
            @test_throws ArgumentError save_model(save_path, ps, st; force = false)

            # With force=true should succeed
            save_model(save_path, ps, st; force = true)
            @test isdir(save_path)
        end
    end

    @testset "In-memory Zarr store (DictStore)" begin
        model = Dense(3 => 2)
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 3, 2)
        y, _ = model(x, ps, st)

        mem_store = isdefined(Zarr, :DictStore) ? Zarr.DictStore() : (isdefined(Zarr, :MemoryStore) ? Zarr.MemoryStore() : Dict{String, Vector{UInt8}}())
        save_model(mem_store, ps, st; model = model)

        lazy_model = load_model(mem_store, model)
        y_loaded, _ = lazy_model(x)
        @test y ≈ y_loaded
    end

    @testset "Convolutional Models (4D Tensors)" begin
        model = Chain(
            Conv((3, 3), 1 => 4, relu),
            Dense(16 => 2)
        )
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 6, 6, 1, 2)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "conv_model.zarr")
            save_model(save_path, ps, st; model = model)

            lazy_model = load_model(save_path)
            @test lazy_model isa LazyLuxModel
            @test lazy_model.ps.layer_1.weight == ps.layer_1.weight
            @test lazy_model.ps.layer_1.bias == ps.layer_1.bias
        end
    end

    @testset "Custom Metadata and Chunking" begin
        model = Dense(4 => 4)
        ps, st = Lux.setup(rng, model)

        custom_meta = Dict{String, Any}(
            "experiment" => "lux_benchmark",
            "epoch" => 42,
            "loss" => 0.0012f0,
        )

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "custom_meta.zarr")
            save_model(save_path, ps, st; model = model, metadata = custom_meta, chunks = (2, 2))

            meta = load_model(save_path; metadata_only = true)
            @test meta["user_metadata"]["experiment"] == "lux_benchmark"
            @test meta["user_metadata"]["epoch"] == 42
            @test isapprox(meta["user_metadata"]["loss"], 0.0012f0; atol = 1.0e-5)

            lazy_model = load_model(save_path, model)
            @test lazy_model.ps.weight == ps.weight
        end
    end

    @testset "Diverse Activation Functions and Model Reconstruction" begin
        for act in (gelu, tanh, sigmoid, Lux.NNlib.swish, Lux.NNlib.leakyrelu)
            model = Chain(Dense(2 => 3, act), Dense(3 => 1))
            ps, st = Lux.setup(rng, model)
            x = randn(rng, Float32, 2, 2)
            y, _ = model(x, ps, st)

            mktempdir() do tmp_dir
                save_path = joinpath(tmp_dir, "act_model.zarr")
                save_model(save_path, ps, st; model = model)

                lazy_model = load_model(save_path)
                @test lazy_model isa LazyLuxModel
                y_rec, _ = lazy_model(x)
                @test y ≈ y_rec
            end
        end
    end

    @testset "Scalar Deserialization & KeyPath Helpers" begin
        kp_empty = LuxZarr.KeyPath()
        @test LuxZarr._keypath_to_path(kp_empty) == ""
        @test LuxZarr._serialize_keypath(kp_empty) == []

        kp = LuxZarr.KeyPath(:layer_1, 2, :weight)
        @test LuxZarr._keypath_to_path(kp) == "layer_1/2/weight"
        serialized = LuxZarr._serialize_keypath(kp)
        @test serialized == [":layer_1", 2, ":weight"]
        deserialized = LuxZarr._deserialize_keypath(serialized)
        @test deserialized.keys == (:layer_1, 2, :weight)

        # Scalar deserialization
        @test LuxZarr._deserialize_scalar(Dict("__type__" => "Symbol", "value" => "mode"), Symbol) == :mode
        @test LuxZarr._deserialize_scalar(Dict("__type__" => "Val", "value" => true), Any) == Val(true)
        @test LuxZarr._deserialize_scalar("Val{true}", Any) == Val(true)
        @test LuxZarr._deserialize_scalar(":test_sym", Symbol) == :test_sym
        @test LuxZarr._deserialize_scalar(42, Float32) === 42.0f0

        # Chunk computation dispatch
        arr = randn(Float32, 4, 8)
        @test LuxZarr._compute_chunks(nothing, arr) == (4, 8)
        @test LuxZarr._compute_chunks(a -> (2, 2), arr) == (2, 2)
        @test LuxZarr._compute_chunks((2, 2), arr) == (2, 2)
        @test LuxZarr._compute_chunks((10, 10), arr) == (4, 8)
    end

    @testset "WrappedFunction, Dropout, and NoOpLayer Serialization" begin
        model = Chain(
            Dense(4 => 4),
            WrappedFunction(relu),
            Dropout(0.2f0),
            NoOpLayer(),
            Dense(4 => 2)
        )
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 3)
        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "special_layers.zarr")
            save_model(save_path, ps, st; model = model)

            lazy_model = load_model(save_path)
            @test lazy_model isa LazyLuxModel
            @test lazy_model.model isa Chain
            @test lazy_model.model.layers[2] isa WrappedFunction
            @test lazy_model.model.layers[3] isa Dropout
            @test lazy_model.model.layers[4] isa NoOpLayer
        end
    end

    @testset "NamedTuple of Models" begin
        models = (
            encoder = Chain(Dense(4 => 8, relu), Dense(8 => 2)),
            decoder = Chain(Dense(2 => 8, relu), Dense(8 => 4)),
        )
        ps_enc, st_enc = Lux.setup(rng, models.encoder)
        ps_dec, st_dec = Lux.setup(rng, models.decoder)
        ps = (encoder = ps_enc, decoder = ps_dec)
        st = (encoder = st_enc, decoder = st_dec)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "nt_models.zarr")
            save_model(save_path, ps, st; model = models)

            # Standalone load
            lazy_nt = load_model(save_path)
            @test lazy_nt isa LazyLuxModel
            @test keys(lazy_nt.model) === (:encoder, :decoder)
            @test lazy_nt.model.encoder isa Chain
            @test lazy_nt.model.decoder isa Chain
            @test lazy_nt.ps.encoder.layer_1.weight == ps.encoder.layer_1.weight
            @test lazy_nt.ps.decoder.layer_1.weight == ps.decoder.layer_1.weight

            # Guided load with model NamedTuple
            lazy_guided = load_model(save_path, models)
            @test lazy_guided isa LazyLuxModel
            @test keys(lazy_guided.model) === (:encoder, :decoder)
        end
    end

    @testset "Faithful Reconstruction and Error Handling" begin
        # 1. Softplus and other activations
        model_softplus = Chain(Dense(3 => 4, softplus), Dense(4 => 2))
        ps_sp, st_sp = Lux.setup(rng, model_softplus)
        x_sp = randn(rng, Float32, 3, 2)
        y_sp, _ = model_softplus(x_sp, ps_sp, st_sp)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "softplus_model.zarr")
            save_model(save_path, ps_sp, st_sp; model = model_softplus)

            lazy_sp = load_model(save_path)
            @test lazy_sp isa LazyLuxModel
            @test lazy_sp.model.layers[1].activation === Lux.NNlib.softplus
            y_sp_loaded, _ = lazy_sp(x_sp)
            @test y_sp ≈ y_sp_loaded
        end

        # 2. Parallel with vcat
        model_parallel_vcat = Parallel(vcat, Dense(3 => 2), Dense(3 => 2))
        ps_pv, st_pv = Lux.setup(rng, model_parallel_vcat)
        x_pv = randn(rng, Float32, 3, 2)
        y_pv, _ = model_parallel_vcat(x_pv, ps_pv, st_pv)
        @test size(y_pv) == (4, 2)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "parallel_vcat.zarr")
            save_model(save_path, ps_pv, st_pv; model = model_parallel_vcat)

            lazy_pv = load_model(save_path)
            @test lazy_pv isa LazyLuxModel
            @test lazy_pv.model.connection === vcat
            y_pv_loaded, _ = lazy_pv(x_pv)
            @test size(y_pv_loaded) == (4, 2)
            @test y_pv ≈ y_pv_loaded
        end

        # 3. SkipConnection with multiplication
        model_skip = SkipConnection(Dense(3 => 3), *)
        ps_sk, st_sk = Lux.setup(rng, model_skip)
        x_sk = randn(rng, Float32, 3, 3)
        y_sk, _ = model_skip(x_sk, ps_sk, st_sk)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "skip_mul.zarr")
            save_model(save_path, ps_sk, st_sk; model = model_skip)

            lazy_sk = load_model(save_path)
            @test lazy_sk isa LazyLuxModel
            @test lazy_sk.model.connection === *
            y_sk_loaded, _ = lazy_sk(x_sk)
            @test y_sk ≈ y_sk_loaded
        end

        # 4. BatchNorm non-default parameters
        bn_custom = BatchNorm(4, relu; epsilon = 1.0f-3, momentum = 0.2f0, affine = false, track_stats = false)
        ps_bn, st_bn = Lux.setup(rng, bn_custom)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "bn_custom.zarr")
            save_model(save_path, ps_bn, st_bn; model = bn_custom)

            lazy_bn = load_model(save_path)
            @test lazy_bn isa LazyLuxModel
            @test lazy_bn.model.epsilon ≈ 1.0f-3
            @test lazy_bn.model.momentum ≈ 0.2f0
            @test lazy_bn.model.affine == false
            @test lazy_bn.model.track_stats == false
            @test lazy_bn.model.activation === Lux.NNlib.relu
        end

        # 5. Error on unknown activation during standalone reconstruction
        info_bad_act = Dict{String, Any}(
            "type" => "Dense",
            "in_dims" => 3,
            "out_dims" => 2,
            "activation" => "my_unregistered_activation_fn",
            "use_bias" => true,
        )
        @test_throws ArgumentError reconstruct_model_from_info(info_bad_act)

        # 6. Error on unknown layer type
        info_unknown_layer = Dict{String, Any}("type" => "CompletelyUnknownLayerType123")
        @test_throws ArgumentError reconstruct_model_from_info(info_unknown_layer)
    end

    @testset "kwargs in load_model and custom reconstruction" begin
        struct CustomLayer <: LuxCore.AbstractLuxLayer end
        LuxZarr.extract_model_info(::CustomLayer) = Dict{String, Any}("type" => "CustomLayer")
        function LuxZarr.reconstruct_layer(::Val{:CustomLayer}, info::AbstractDict; custom_opt = false, kwargs...)
            custom_opt ? CustomLayer() : throw(ArgumentError("custom_opt required"))
        end

        model = CustomLayer()
        ps, st = NamedTuple(), NamedTuple()
        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "custom_layer.zarr")
            save_model(save_path, ps, st; model = model)

            # Without custom_opt=true, reconstruction throws ArgumentError
            @test_throws ArgumentError load_model(save_path)

            # With custom_opt=true forwarded via kwargs...
            lazy_loaded = load_model(save_path; custom_opt = true)
            @test lazy_loaded isa LazyLuxModel
            @test lazy_loaded.model isa CustomLayer
        end
    end

    @testset "resolve_activation, resolve_connection, and Custom Layer Hook" begin
        @test resolve_activation("relu") === Lux.NNlib.relu
        @test resolve_activation("identity") === identity
        @test resolve_activation("tanh") === Lux.NNlib.tanh_fast
        @test_throws ArgumentError resolve_activation("unsupported_activation_xyz")

        @test resolve_connection("+") === +
        @test resolve_connection("*") === *
        @test resolve_connection("vcat") === vcat
        @test_throws ArgumentError resolve_connection("unsupported_connection_xyz")

        struct MyLinearLayer{F} <: LuxCore.AbstractLuxLayer
            in_dims::Int
            out_dims::Int
            activation::F
        end
        Lux.initialparameters(rng::Random.AbstractRNG, l::MyLinearLayer) = (weight = randn(rng, Float32, l.out_dims, l.in_dims),)
        Lux.initialstates(::Random.AbstractRNG, ::MyLinearLayer) = NamedTuple()
        (l::MyLinearLayer)(x, ps, st) = (l.activation.(ps.weight * x), st)

        LuxZarr.extract_model_info(l::MyLinearLayer) = Dict{String, Any}(
            "type" => "MyLinearLayer",
            "in_dims" => l.in_dims,
            "out_dims" => l.out_dims,
            "activation" => string(l.activation),
        )
        function LuxZarr.reconstruct_layer(::Val{:MyLinearLayer}, info::AbstractDict; kwargs...)
            act = resolve_activation(info["activation"])
            return MyLinearLayer(Int(info["in_dims"]), Int(info["out_dims"]), act)
        end

        custom_m = MyLinearLayer(5, 3, relu)
        ps_c, st_c = Lux.setup(rng, custom_m)
        x_c = randn(rng, Float32, 5, 2)
        y_expected, _ = custom_m(x_c, ps_c, st_c)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "mylinear.zarr")
            save_model(save_path, ps_c, st_c; model = custom_m)

            # Lazy standalone load and evaluation
            lazy_custom = load_model(save_path)
            @test lazy_custom isa LazyLuxModel
            @test lazy_custom.model.in_dims == 5
            @test lazy_custom.model.out_dims == 3
            @test lazy_custom.model.activation === Lux.NNlib.relu
            y_lazy, _ = lazy_custom(x_c)
            @test y_lazy ≈ y_expected

            # Eager standalone load
            ps_eager, st_eager, m_eager = load_model(save_path; lazy = false)
            @test m_eager isa MyLinearLayer
            @test m_eager.in_dims == 5
            @test ps_eager.weight == ps_c.weight
        end
    end
end
