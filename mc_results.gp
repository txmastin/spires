# mc_results.gp
set title "Memory Capacity vs Alpha"
set xlabel "Alpha"
set ylabel "MC"
set key top right
set grid

plot "data/alpha_mc.dat" using 1:4 with linespoints title "MC"


pause -1
