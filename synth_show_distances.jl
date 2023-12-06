using CSV, DataFrames, Statistics

df = CSV.read("data/expresults/bc_td_distances/synth_bcs_div_20.csv", DataFrame)

using StatsPlots

min, max = extrema(df.distance)

df.ms = round.(map(d -> 1 + (d / (max - min) * 9), df.distance))

gfs = groupby(df, :output)

# synth_func(x) = (x+2)*(x^2-4)
# x = range(-10, 10, 1_000)
# y = synth_func.(x)

#p_gt = plot(df.x1, df.x2, xlims=(-10, 10), ylims=(-11,10), seriestype=:scatter, markersize=.5)
# have legend in upper left corner

p_gt = plot(xlims=(-5, 4), ylims=(-11,10), seriestype=:scatter, linestyle=:none, markersize=.5, legend=false)
title!(p_gt, "Distance to nearest training data point")
xlabel!(p_gt, "argument 1 (x)")
ylabel!(p_gt, "argument 2 (y)")

df_p = filter(r -> r.output == 1, eachrow(df)) |> DataFrame
df_m = filter(r -> r.output == 0, eachrow(df)) |> DataFrame

df_p_larger = df_p[filter(i -> df_p[i, :distance] > df_m[i, :distance], 1:nrow(df_p)), :]
df_m_larger = df_m[filter(i -> df_m[i, :distance] > df_p[i, :distance], 1:nrow(df_p)), :]
df_p_smaller = df_p[filter(i -> df_p[i, :distance] ≤ df_m[i, :distance], 1:nrow(df_p)), :]
df_m_smaller = df_m[filter(i -> df_m[i, :distance] ≤ df_p[i, :distance], 1:nrow(df_p)), :]

plot!(df_p_larger.x1, df_p_larger.x2, markersize=df_p_larger.ms, color=:green, marker=:circle, seriestype=:scatter, markerstrokewidth=0, label = false)
plot!(df_m_larger.x1, df_m_larger.x2, markersize=df_m_larger.ms, color=:red, markerstrokewidth=0, marker=:circle, seriestype=:scatter, label = false)
plot!(df_m_smaller.x1, df_m_smaller.x2, markersize=df_m_smaller.ms, color=:red, markerstrokewidth=0, marker=:circle, seriestype=:scatter, label = "invalid-side", legend=:topleft)
plot!(df_p_smaller.x1, df_p_smaller.x2, markersize=df_p_smaller.ms, color=:green, marker=:circle, seriestype=:scatter, markerstrokewidth=0, label = "valid-side")

p_gt