# Gnuplot script to visualize reservoir computer results
# This version displays both plots simultaneously in a single window.

# --- Multiplot Setup ---
# Set a layout of 2 rows and 1 column to stack the plots vertically.
# The 'title' for the whole window is set here.
set multiplot layout 2,1 title "Reservoir Simulation Results"

# --- PLOT 1: Error Trace (Top Plot) ---
set title "Training: Average Error over Epochs"
set xlabel "Epoch"
set ylabel "Average Error"
set grid
set logscale x
plot 'error_trace.dat' using 1:2 with lines title 'Average Error' lc rgb "#0072B2"

unset logscale x

# --- PLOT 2: Final Signal Comparison (Bottom Plot) ---
set title "Signal Reconstruction after Training"
set xlabel "Timestep"
set ylabel "Amplitude"
set grid
set key top right
plot 'output_signals.dat' using 1:2 with lines title 'Original Signal' lw 2 lc rgb "#009E73", \
     ''                   using 1:3 with lines title 'Reservoir Output' lw 2 lc rgb "#D55E00"


# --- End Multiplot ---
# Unset multiplot mode to finalize the plot.
unset multiplot

pause -1

