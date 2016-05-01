julia Test.jl 2 2>&1 | tee ../experiments/test_02_scenes_output.txt
julia Test.jl 10 2>&1 | tee ../experiments/test_10_scenes_output.txt
julia Test.jl 654 2>&1 | tee ../experiments/test_all_scenes_output.txt
