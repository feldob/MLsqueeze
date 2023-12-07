
# preprocess the input spaces into continuous spaces
include("mlboundaries_pp.jl")

# create BC's for all datasets with 10 and 20 candidates, for search with and without diversity in search
include("mlboundaries_all.jl")

# distance table in LaTeX
# + normalized distance plots 
include("mlboundaries_distances_models.jl")

# extract distances for sync dataset for 20 candidates
include("mlboundaries_distances_to_training_data.jl")