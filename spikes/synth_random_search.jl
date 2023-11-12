include("random_search.jl")

using CSV, DataFrames, DecisionTree

DataDir = joinpath(@__DIR__(), "..", "data")
DF = CSV.read(joinpath(DataDir, "synth.csv"), DataFrame)

# Inputs... then output
O = "class"
oidx = findfirst(==(O), names(DF))
Inps = names(DF)[1:(oidx-1)]

SearchRangePair, Points = config_search(DF)

# fitness function: the format is floats... four floats, class is a consequence only.
synth_func(x) = (x+2)*(x^2-4)

# train a tree on depth
function plantatree(df::DataFrame, depth=3)
    model = DecisionTreeClassifier(max_depth=depth)
    training = hcat(df.x1, df.x2)
    return DecisionTree.fit!(model, training, df.class)
end

poormodel = plantatree(DF)
bettermodel = plantatree(DF, 10)
evenbettermodel = plantatree(DF, 20)

check_synth_valid(x::Number,y::Number) = y > synth_func(x) ? 1 : 0
function ml_check_synth_valid(x::Number,y::Number)
    #return DecisionTree.predict(poormodel, [x,y])
    #return DecisionTree.predict(bettermodel, [x,y])
    return DecisionTree.predict(evenbettermodel, [x,y])
end

cands = diverse_bcs(pd_fitness(ml_check_synth_valid), Points, 500, 20, SearchRangePair)

using StatsPlots

as = map(c -> (c[1], c[2]), cands)
bs = map(c -> (c[3], c[4]), cands)
plot(as, seriestype=:scatter)
plot!(bs, seriestype=:scatter)