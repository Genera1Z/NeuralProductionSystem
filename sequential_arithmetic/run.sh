for seed in 20230104 202301 2023 20 1
do
  for share in 1000 1100 1110 1101 1111
  do
    for goal in NormalizedMse MSE
    do
      python main.py $seed $share $goal
    done
  done
done
