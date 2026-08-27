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
            save_model(save_path, ps, st; metadata=Dict("custom_tag" => "pure_tree"))

            # Standalone loading
            ps_loaded, st_loaded = load_model(save_path)
            @test ps_loaded.layer_1.weight == ps.layer_1.weight
            @test ps_loaded.layer_1.bias == ps.layer_1.bias
            @test ps_loaded.layer_2.weight == ps.layer_2.weight
            @test ps_loaded.layer_2.bias == ps.layer_2.bias
            @test st_loaded.step == 100
            @test st_loaded.active == Val(true)
            @test st_loaded.mode == :eval

            # Guided loading with ps, st
            ps_guided, st_guided = load_model(save_path, ps, st)
            @test ps_guided.layer_1.weight == ps.layer_1.weight
            @test st_guided.step == 100
        end
    end

    @testset "Dense and Chain Models" begin
        model = Chain(
            Dense(4 => 8, relu),
            Dense(8 => 2)
        )
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 3)
        y, _ = model(x, ps, st)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "dense_chain.zarr")

            # Save model
            save_model(save_path, ps, st; model=model, metadata=Dict("tag" => "test_dense"))
            @test isdir(save_path)

            # Metadata only inspection
            meta = load_model(save_path; metadata_only=true)
            @test meta["format"] == "lux_zarr"
            @test meta["user_metadata"]["tag"] == "test_dense"
            @test haskey(meta, "model_info")
            @test meta["model_info"]["type"] == "Chain"

            # Guided load with model
            ps_loaded, st_loaded = load_model(save_path, model)
            y_loaded, _ = model(x, ps_loaded, st_loaded)
            @test y ≈ y_loaded

            # Guided load with (ps, st)
            ps_loaded2, st_loaded2 = load_model(save_path, ps, st)
            y_loaded2, _ = model(x, ps_loaded2, st_loaded2)
            @test y ≈ y_loaded2

            # Standalone load (reconstruction)
            ps_standalone, st_standalone = load_model(save_path)
            y_standalone, _ = model(x, ps_standalone, st_standalone)
            @test y ≈ y_standalone
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
            save_model(save_path, ps, st; model=model)

            # Guided load with model
            ps_loaded, st_loaded = load_model(save_path, model)
            @test st_loaded.layer_2.running_mean ≈ st.layer_2.running_mean
            @test st_loaded.layer_2.running_var ≈ st.layer_2.running_var
            @test st_loaded.layer_2.training == st.layer_2.training

            # Standalone load
            ps_standalone, st_standalone = load_model(save_path)
            @test ps_standalone.layer_1.weight ≈ ps.layer_1.weight
            @test st_standalone.layer_2.running_mean ≈ st.layer_2.running_mean
            @test st_standalone.layer_2.running_var ≈ st.layer_2.running_var
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
            save_model(save_path, ps, st; model=model_valid)

            # Guided load
            ps_loaded, st_loaded = load_model(save_path, model_valid)
            y_loaded, _ = model_valid(x, ps_loaded, st_loaded)
            @test y ≈ y_loaded

            # Standalone load
            ps_standalone, st_standalone = load_model(save_path)
            y_standalone, _ = model_valid(x, ps_standalone, st_standalone)
            @test y ≈ y_standalone
        end
    end

    @testset "Overwrite / force handling" begin
        model = Dense(2 => 2)
        ps, st = Lux.setup(rng, model)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "test_force.zarr")
            save_model(save_path, ps, st)

            # Without force=true on existing directory should error
            @test_throws ArgumentError save_model(save_path, ps, st; force=false)

            # With force=true should succeed
            save_model(save_path, ps, st; force=true)
            @test isdir(save_path)
        end
    end

    @testset "In-memory Zarr store (DictStore)" begin
        model = Dense(3 => 2)
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 3, 2)
        y, _ = model(x, ps, st)

        mem_store = isdefined(Zarr, :DictStore) ? Zarr.DictStore() : (isdefined(Zarr, :MemoryStore) ? Zarr.MemoryStore() : Dict{String, Vector{UInt8}}())
        save_model(mem_store, ps, st; model=model)

        ps_loaded, st_loaded = load_model(mem_store, model)
        y_loaded, _ = model(x, ps_loaded, st_loaded)
        @test y ≈ y_loaded
    end

    @testset "Convolutional Models (4D Tensors)" begin
        model = Chain(
            Conv((3, 3), 1 => 4, relu),
            Dense(16 => 2)
        )
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 6, 6, 1, 2)
        conv_out, _ = model.layers.layer_1(x, ps.layer_1, st.layer_1)

        mktempdir() do tmp_dir
            save_path = joinpath(tmp_dir, "conv_model.zarr")
            save_model(save_path, ps, st; model=model)

            # Guided load
            ps_loaded, st_loaded = load_model(save_path, model)
            @test ps_loaded.layer_1.weight == ps.layer_1.weight
            @test ps_loaded.layer_1.bias == ps.layer_1.bias

            # Standalone load
            ps_standalone, st_standalone, model_standalone = load_model(save_path)
            @test ps_standalone.layer_1.weight == ps.layer_1.weight
            @test model_standalone isa Chain
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
            save_model(save_path, ps, st; model=model, metadata=custom_meta, chunks=(2, 2))

            meta = load_model(save_path; metadata_only=true)
            @test meta["user_metadata"]["experiment"] == "lux_benchmark"
            @test meta["user_metadata"]["epoch"] == 42
            @test isapprox(meta["user_metadata"]["loss"], 0.0012f0; atol=1e-5)

            ps_loaded, _ = load_model(save_path, model)
            @test ps_loaded.weight == ps.weight
        end
    end
end
