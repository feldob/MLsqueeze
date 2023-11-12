@testset "test exposer" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES)
    be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))
    candidates = apply(be; iterations=10, initial_candidates=5)

    @test length(candidates) > 0
    @test all([Distances.evaluate(Euclidean(), left(c), right(c)) < delta(be.bs) for c in candidates])

    df = todataframe(candidates, bmi_classification)
    plots(df, MLsqueeze.ranges(td))
end

@testset "one vs all exposer" begin
    td = TrainingData(bmi_classification; ranges=BMI_RANGES)
    be = BoundaryExposer(td, bmi_classification, BoundarySqueeze(BMI_RANGES))
    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=true)
    df = todataframe(candidates, bmi_classification)

    # TODO have some real tests
    @test true
end

@testset "iris classifier test" begin
    df = CSV.read("../data/Iris.csv", DataFrame)
    
    inputs = [:SepalLengthCm, :SepalWidthCm, :PetalLengthCm, :PetalWidthCm]
    output = :Species
    
    ranges = deriveranges(df, inputs)
    @test length(ranges) == 4
    @test ranges isa Vector{<:Tuple}

    td = TrainingData("iris", df; inputs, output)

    # create a model that can be used for classification
    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)

    be = BoundaryExposer(td, modelsut, BoundarySqueeze(MLsqueeze.ranges(td)))
    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=false)
    df = todataframe(candidates, modelsut; output)

    plots(df, MLsqueeze.ranges(td); output)    
end

@testset "classifier test (categorical and ordinal titanic)" begin
    df = CSV.read("../data/titanic.csv", DataFrame)
    
    # TODO until missing handling not supported, filter out
    filter!(r -> !ismissing(r.Age), df)

    inputs = [:Pclass, :Age]
    output = :Survived
    
    ranges = deriveranges(df, inputs)

    td = TrainingData("titanic", df; inputs, output)

    # TODO do even for categorical, such as :Sex

    modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=3), fit=DecisionTree.fit!)

    be = BoundaryExposer(td, modelsut, BoundarySqueeze(MLsqueeze.ranges(td)))
    candidates = apply(be; iterations=10, initial_candidates=5, one_vs_all=false)
    df = todataframe(candidates, modelsut; output)

    plots(df, MLsqueeze.ranges(td); output)    
end

