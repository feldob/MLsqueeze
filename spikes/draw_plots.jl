using DataFrames, CSV

df_synth = CSV.read("data/synth.csv", DataFrame)
df_synthb = CSV.read("data/synth_bound.csv", DataFrame)
df_bmi = CSV.read("data/bmi.csv", DataFrame)