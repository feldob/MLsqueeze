using CSV, DataFrames

iterations = 2000
initial_candidates = 10

output = :class
ranges = [(-12,12), (-12,12)]

using MLsqueeze

td = TrainingData(check_synth_valid; ranges)

using DecisionTree # with model

# 2a) diversity
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
be = BoundaryExposer(td, modelsut) # instantiate search alg
candidates = apply(be; iterations, initial_candidates, optimizefordiversity=false) # search and collect candidates

df_model_diversity = todataframe(candidates, modelsut; output)

using StatsPlots

x = range(-10, 10, 1_000)
y = synth_func.(x)

p_model = plot(x, y, xlims=(-10, 10), ylims=(-11, 10), legend=false)
xlabel!(p_model, "argument 1 (x)")
ylabel!(p_model, "argument 2 (y)")

p_div_model = deepcopy(p_model)
title!(p_div_model, "decision tree boundary")

sort!(df_model_diversity, [:y, :x])

x_r = df_model_diversity.x
y_r = df_model_diversity.y

plot!(p_div_model, x_r, y_r, seriestype=:scatter, markersize=2, color=:black)

CSV.write("data/synth_model_points.csv", df_model_diversity)
png(p_div_model, "data/synth_total_model")