using CSV, DataFrames

# ----- adult

df_a = CSV.read("data/adult.csv", DataFrame)

df_a.gender = map(g -> g == "Male" ? 0.0 : 1.0, df_a.gender)

CSV.write("data/adult_pp.csv", df_a)

# ----- titanic

df_t = CSV.read("data/titanic.csv", DataFrame)

filter!(r -> !ismissing(r.Age), df_t)
df_t.Age = float.(df_t.Age)

filter!(r -> !ismissing(r.Pclass), df_t)
df_t.Pclass = float.(df_t.Pclass)

df_t.Sex = map(d -> d == "male" ? 0.0 : 1.0, df_t.Sex)
df_t.Sex = float.(df_t.Sex)

CSV.write("data/titanic_pp.csv", df_t)

# ----- student data

df_s = CSV.read("data/student_data.csv", DataFrame)

df_s.sex = map(d -> d == "M" ? 0.0 : 1.0, df_s.sex)
df_s.school = map(d -> d == "GP" ? 0.0 : 1.0, df_s.school)
df_s.address = map(d -> d == "U" ? 0.0 : 1.0, df_s.address)
df_s.famsize = map(d -> d == "LE3" ? 0.0 : 1.0, df_s.famsize)
df_s.Pstatus = map(d -> d == "A" ? 0.0 : 1.0, df_s.Pstatus)

df_s.schoolsup = map(d -> d == "no" ? 0.0 : 1.0, df_s.schoolsup)
df_s.famsup = map(d -> d == "no" ? 0.0 : 1.0, df_s.famsup)
df_s.paid = map(d -> d == "no" ? 0.0 : 1.0, df_s.paid)
df_s.activities = map(d -> d == "no" ? 0.0 : 1.0, df_s.activities)
df_s.nursery = map(d -> d == "no" ? 0.0 : 1.0, df_s.nursery)
df_s.higher = map(d -> d == "no" ? 0.0 : 1.0, df_s.higher)
df_s.internet = map(d -> d == "no" ? 0.0 : 1.0, df_s.internet)
df_s.romantic = map(d -> d == "no" ? 0.0 : 1.0, df_s.romantic)

CSV.write("data/student_data_pp.csv", df_s)

# ----- car evaluation

df_c = CSV.read("data/car_evaluation.csv", DataFrame)

df_c.buyingprice = map(d -> d == "low" ? 0.0 : d == "med" ? 1.0 : d == "high" ? 2.0 : 3.0, df_c.buyingprice)
df_c.maintenancecost = map(d -> d == "low" ? 0.0 : d == "med" ? 1.0 : d == "high" ? 2.0 : 3.0, df_c.maintenancecost)
df_c.doors = map(d -> d == "2" ? 0.0 : d == "3" ? 1.0 : d == "4" ? 2.0 : 3.0, df_c.doors)
df_c.capacity = map(d -> d == "2" ? 1.0 : d == "4" ? 2.0 : 3.0, df_c.capacity)
df_c.luggageboot = map(d -> d == "small" ? 0.0 : d == "med" ? 1.0 : 2.0, df_c.luggageboot)
df_c.safety = map(d -> d == "low" ? 0.0 : d == "med" ? 1.0 : 2.0, df_c.safety)

CSV.write("data/car_evaluation_pp.csv", df_c)
