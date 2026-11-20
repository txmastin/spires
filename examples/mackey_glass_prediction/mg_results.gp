# Gnuplot script to visualize Mackey-Glass one-step-ahead prediction.
# Columns of data/output_signals.dat: t, target x(t+1), prediction, is_test.

set title "Mackey-Glass one-step-ahead prediction (held-out test segment)"
set xlabel "Timestep"
set ylabel "x(t+1)"
set grid
set key top right
plot 'data/output_signals.dat' using 1:($4 == 1 ? $2 : 1/0) with lines title 'Target' lw 2 lc rgb "#009E73", \
     ''                        using 1:($4 == 1 ? $3 : 1/0) with lines title 'Reservoir prediction' lw 2 lc rgb "#D55E00"

pause -1
