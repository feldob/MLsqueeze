# synthetic boundary described as bounding box built out of non-linear functions
synth_bound_func_left(x) = 5 + ((x+12)+2)*((x+12)^2-4)
synth_bound_func_right(x) = -5 + ((x-12)+2)*((x-12)^2-4)
synth_bound_func_up(x) = 10
synth_bound_func_down(x) = -17

function check_synth_bound_valid(x::Number,y::Number)
    if y > synth_bound_func_up(x) ||
        y < synth_bound_func_down(x) ||
        y > synth_bound_func_left(x) ||
        y < synth_bound_func_right(x)
        return false
    end
    return true
end

x = range(-20, 20, 1_000)
y_right = synth_bound_func_right.(x)
y_left = synth_bound_func_left.(x)
y_up = synth_bound_func_up.(x)
y_down = synth_bound_func_down.(x)

using StatsPlots

lims = (-20, 20)

plot(x, y_right, xlims=lims, ylims=lims, label = "boundary (right)")
plot!(x, y_left, label = "boundary (left)")
plot!(x, y_up, label = "boundary (up)")
plot!(x, y_down, label = "boundary (down)")

randinrange(n::Integer) = -20 .+ rand(n) .* 40

using DataFrames

npoints = 1000
df = DataFrame()
df.x1 = randinrange(npoints)
df.x2 = randinrange(npoints)

df.class = zeros(Int, npoints)
foreach(r -> check_synth_bound_valid(r.x1, r.x2) ? r.class = 1 : nothing, eachrow(df))

gfs = groupby(df, :class)
plot!(gfs[1].x1,gfs[1].x2, seriestype=:scatter, label = "invalid")
plot!(gfs[2].x1,gfs[2].x2, seriestype=:scatter, label = "valid")

## to print plot
png("data/synth_bound_class.png")

using DecisionTree

for depth in 2:5
    model = DecisionTreeClassifier(max_depth=depth)
    training = hcat(df.x1, df.x2)
    test = training # OBS here using same data for training and testing
    fit!(model, training, df.class)
    df[!, "DT_d$(depth)"]=predict(model, test)
end


## to export to CSV file
using CSV
CSV.write("data/synth_bound.csv", df)
