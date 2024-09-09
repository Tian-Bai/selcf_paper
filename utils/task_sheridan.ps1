cd "D:\Github\selcf_paper\utils"

$dataset = "HIVINT"
$sample = "1.0"

for ($i = 1; $i -le 100; $i++) {
    python Sheridan15-vs-Conformal.py $dataset $sample $i
}