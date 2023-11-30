using CSV, DataFrames

df_model_random_10 = CSV.read("data/synth_model_points_random_10.csv", DataFrame)
df_model_random_20 = CSV.read("data/synth_model_points_random_20.csv", DataFrame)
df_model_diversity_10 = CSV.read("data/synth_model_points_div_10.csv", DataFrame)
df_model_diversity_20 = CSV.read("data/synth_model_points_div_20.csv", DataFrame)

using StatsPlots

synth_func(x) = (x+2)*(x^2-4)

x = range(-12, 12, 1000)
y = synth_func.(x)

p_model = plot(x, y, xlims=(-10.5, 10.5), ylims=(-10.5, 10.5), seriestype=:line, legend=false)
xlabel!(p_model, "argument 1 (x)")
ylabel!(p_model, "argument 2 (y)")

plot!(p_model, td.df.x, td.df.y, seriestype=:scatter, markershape=:xcross, color=:gray)

plot!(p_model, df_high_res.x, df_high_res.y, color=:black, markersize=2, seriestype=:scatter)

# plots
function plotbcs(df, p_model, title, filename, color)
    p_model_c = deepcopy(p_model)
    plot!(p_model_c, df.x, df.y, seriestype=:scatter, marker=:square, color=color)
    title!(p_model_c, title)
    png(p_model_c, filename)
end

plotbcs(df_model_random_10, p_model, "Boundary Candidates (no diversity in search)", "data/synth_random_model_10", :lightgreen)
plotbcs(df_model_diversity_10, p_model, "Boundary Candidates (diversity in search)", "data/synth_div_model_10", :red)
plotbcs(df_model_random_20, p_model, "Boundary Candidates (no diversity in search)", "data/synth_random_model_20", :lightgreen)
plotbcs(df_model_diversity_20, p_model, "Boundary Candidates (diversity in search)", "data/synth_div_model_20", :red)
