$q_LIST = @(1, 2, 5)
for ($sig = 1; $sig -le 10; $sig++) {
    for ($nt_id = 1; $nt_id -le 4; $nt_id++) {
        for ($set_id = 1; $set_id -le 8; $set_id++) {
            foreach ($q in $q_LIST) {
                for ($seed = 1; $seed -le 100; $seed++) {
                    python "D:\Github\selcf_paper\utils\simu.py" $sig $nt_id $set_id $q $seed
                }
            }
        }
    }
}
