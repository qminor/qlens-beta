unset key

set size ratio -1
set cbrange [-3:3]
set sample 30000
set xrange [:] noextend; set yrange [:] noextend
plot [-1.75:1.75][-1.75:1.75] 'chains_delaunay_fit/img_pixel.dat' u (.035*$1+-1.7325):(.035*$2+-1.7325):3 matrix w image, 'chains_delaunay_fit/crit.dat' w lines lw 1 lc 0 t 'critical curves'