for d in 2 4 8 16 24; do
  DEPTH=$d NT=4 python3 train.py deep 15 deep$d.pt 0 1 256 > log_deep$d.txt 2>&1
done
for d in 2 4 8 16 24; do
  for dec in none perm; do
    D=$dec; [ "$dec" = none ] && D=
    DEPTH=$d RN=1 DECOR=$D NT=4 python3 eval_sc.py deep deep$d.pt 1000 > ev_deep${d}_$dec.txt 2>&1
  done
done
echo SWEEPDONE
