# synthetic boundary described as non-linear function
synth_func(x) = (x+2)*(x^2-4)

check_synth_valid(x::Number,y::Number) = y > synth_func(x)

x = range(-10, 10, 1_000)
y = synth_func.(x)

using Plots

plot(x, y, xlims=(-10, 10), ylims=(-10,10), label = "boundary")

randinrange(n::Integer) = -10 .+ rand(n) .* 20

using DataFrames

npoints = 200
df = DataFrame()
df.x1 = randinrange(npoints)
df.x2 = randinrange(npoints)

df.class = zeros(Int, npoints)
foreach(r -> check_synth_valid(r.x1, r.x2) ? r.class = 1 : nothing, eachrow(df))

gfs = groupby(df, :class)
plot!(gfs[1].x1,gfs[1].x2, seriestype=:scatter, label = "invalid")
plot!(gfs[2].x1,gfs[2].x2, seriestype=:scatter, label = "valid")

df

## to print plot
# png("data/synth_class.png")

## to export to CSV file
# using CSV
# CSV.write("data/synth.csv", df)