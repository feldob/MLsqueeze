using CSV, DataFrames

DataDir = "/Users/feldt/Companies/OpenData/kaggle/compas/propublicaCompassRecividism_data_fairml.csv/derived"
const DF = CSV.read(joinpath(DataDir, "compas_mothilal.csv"), DataFrame)

O = "Class"
Inps = filter(n -> n != O, names(DF))

const Inputs = Matrix(DF[:, Inps])

# Get upper and lower bounds for searching among the input columns
function get_bounds(df, InputColumns)
    lbs = zeros(Int, length(InputColumns))
    hbs = zeros(Int, length(InputColumns))
    for (i, icol) in enumerate(InputColumns)
        lbs[i] = minimum(df[:, icol])
        hbs[i] = maximum(df[:, icol])
    end
    return lbs, hbs
end

lowerbounds, higherbounds = get_bounds(df, Inps)

# As input distance we just use the Euclidean, for now. This is probably not good
# in the general case but we just want to get going.
using Distances
const IDist = Euclidean()

# Get the distance between two rows in the data frame
function df_row_distance(i, j)
    return IDist(Inputs[i, :], Inputs[j, :])
end

df_row_distance(1, 2)
df_row_distance(1, 3)