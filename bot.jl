using DataFrames
using CSV
using GLMakie
using Makie
using Statistics

df = DataFrame(CSV.File("~/IBM2min.csv"))
labels = DataFrame(CSV.File("~/GuessedIBMlabels.csv")).label

close = df.Close

close = last(close, length(labels))
buys = findall(labels .== 1)
sells = findall(labels .== -1)

@. close = /(close - close[1], close[1])

# basic backtesting program to test returns of labels
global account = 0
global long = false
global short = false
global bought_at = NaN
history = Array{Float32}(undef, length(labels))
for i in 1:length(labels)
    if !long && !short && labels[i] == 1
        global bought_at = close[i] #* 1.005 #weight for slippage
        global long = true
    end
    if long && !short && labels[i] == -1
        global account += close[i] - bought_at
        global long = false
        global bought_at = NaN
    end

    if !short && !long && labels[i] == -1
        global bought_at = close[i] #* 0.995 #weight for slippage
        global short = true
    end
    if short && !long && labels[i] == 1
        global account += bought_at-close[i]
        global short = false
        global bought_at = NaN
    end
    
    history[i] = account
end



# Plotting a bunch of stuff

fig = Figure()
size = 5
lines(fig[1, 1], close)
lines!(fig[1, 1], history, color="cyan")
scatter!(fig[1, 1], buys, close[buys], color="green", markersize=size)
scatter!(fig[1, 1], sells, close[sells], color="red", markersize=size)

lines(fig[2, 1], labels, color="black")
println(account)
# Risk free return is set to 0
println(last(history) / std(history))
current_figure()
