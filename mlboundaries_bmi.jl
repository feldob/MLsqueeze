using CSV, DataFrames

iterations = 2000
initial_candidates = 10

output = :class

using MLsqueeze
td = TrainingData(bmi_classification; ranges=BMI_RANGES)
bs = BoundarySqueeze(BMI_RANGES)

# 1a) diversity
be = BoundaryExposer(td, bmi_classification, bs)
candidates = apply(be; iterations, initial_candidates) # search and collect candidates

df_diversity = todataframe(candidates, bmi_classification; output)

# b) random
iterations = 17

be = BoundaryExposer(td, bmi_classification, bs) # instantiate search alg
candidates = apply(be; iterations, initial_candidates, optimizefordiversity=false) # search and collect candidates

df_random = todataframe(candidates, bmi_classification; output)

using StatsPlots

p = plot(xlims=BMI_RANGES[1], ylims=BMI_RANGES[2], legend=false)
xlabel!(p, "argument 2 (weight in kg)")
ylabel!(p, "argument 1 (height in cm)")

p_random = deepcopy(p)
p_div = deepcopy(p)

y_r = df_random.height
x_r = df_random.weight

plot!(p_random, x_r, y_r, seriestype=:scatter, marker=:square, color=:lightgreen)

title!(p_random_gt, "Boundary Candidates (no diversity in search)")

y_r = df_diversity.height
x_r = df_diversity.weight

plot!(p_div, x_r, y_r, seriestype=:scatter, color=:black)

title!(p_div, "Boundary Candidates (diversity in search)")

png(p_div, "data/bmi_diversity")
png(p_random, "data/bmi_random")

p_div2 = plots(df_diversity, MLsqueeze.ranges(td); output)
png(p_div2[1], "data/bmi_diversity2")

p_random2 = plots(df_random, MLsqueeze.ranges(td); output)
png(p_random2[1], "data/bmi_random2")