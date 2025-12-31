set terminal pngcairo enhanced size 800,600 font "Arial,12"
set output 'metrics_finetuned.png'

set title "Model Performance Metrics (Fine-tuned MarianNMT)"
set xlabel "Metrics"
set ylabel "Scores"

set xtics ("BLEU" 1, "chrF" 2, "METEOR" 3, "BERTScore F1" 4) rotate by -45

set yrange [0:100]

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

$DATA << EOD
1 80.48902338868102
2 88.34768749913579
3 81.86412705791294
4 90.02937078475952
EOD

plot $DATA using 1:2 with boxes lc rgb "#4daf4a" notitle


