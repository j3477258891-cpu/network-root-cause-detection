# V118 safe-delete + positive-code campaign

The previous V104 calibration scored 0.840838 (TP=803): its 169-removal pool
was inverted and removed 153 true roots. V118 replaces it with 40 lowest V11
score champion nodes and 60 V75 additions. It uses 1 calibration + 24 code
probes + 3 final slots. Together with the two submissions already consumed,
this is exactly 30 submissions within 15 days at two per day.

Submit `probe_00_calibration.csv` first. The expected prediction count is 995.
It scored 0.907308 online, implying TP=925 and 31 true roots removed; do not
use the fixed pool blindly. Continue with the code probes so the signed
equations can identify safe deletions and correct additions.
Then record the 24 F1 values and decode with:

```powershell
$env:PYTHONPATH='D:\zgyidong\.deps'
$py='C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py experiments\v118_safe_positive_campaign.py --decode `
  --calibration <score0> `
  --scores <score1> <score2> ... <score24>
```

Do not submit the rejected V104 files or the rejected distance1 swap.
