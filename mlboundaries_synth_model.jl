using CSV, DataFrames

iterations = 2000
initial_candidates = 10

output = :class
ranges = [(-10,10), (-10,10)]

using MLsqueeze

td = TrainingData(check_synth_valid; ranges)
bs = BoundarySqueeze(td)

using DecisionTree # with model

# 2a) diversity
modelsut = getmodelsut(td; model=DecisionTree.DecisionTreeClassifier(max_depth=7), fit=DecisionTree.fit!)
be = BoundaryExposer(td, modelsut, bs) # instantiate search alg
candidates = apply(be; iterations, initial_candidates) # search and collect candidates

df_model_diversity = todataframe(candidates, modelsut; output)

# b) random
iterations = 17
be = BoundaryExposer(td, modelsut, bs) # instantiate search alg
candidates = apply(be; iterations, initial_candidates, optimizefordiversity=false) # search and collect candidates

df_model_random = todataframe(candidates, modelsut; output)

x = range(-10, 10, 1_000)
y = synth_func.(x)

using StatsPlots

p_model = plot(x, y, xlims=(-10, 10), ylims=(-11, 10), legend=false, label = "ground truth boundary")
xlabel!(p_model, "argument 1 (x)")
ylabel!(p_model, "argument 2 (y)")

p_random_model = deepcopy(p_model)
p_div_model = deepcopy(p_model)

x_r = df_model_random.x
y_r = df_model_random.y

plot!(p_random_model, x_r, y_r, seriestype=:scatter, marker=:square, color=:lightgreen, label = "random candidates")

title!(p_random_model, "Boundary Candidates (no diversity in search)")

x_r = df_model_diversity.x
y_r = df_model_diversity.y

plot!(p_div_model, x_r, y_r, seriestype=:scatter, color=:black, label = "diverse candidates")

title!(p_div_model, "Boundary Candidates (diversity in search)")

png(p_div_model, "data/synth_diversity_model")
png(p_random_model, "data/synth_random_model")