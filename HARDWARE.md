# Hardware used for the full 50-file run

| Item | Value |
|---|---|
| Where it ran (own computer / university machine / Colab / Kaggle / rented GPU) | Google Colab (free tier) |
| GPU model and memory | NVIDIA Tesla T4, 15 GB (15,360 MiB) |
| CPU model and number of cores / threads | Intel Xeon @ 2.00 GHz, 2 vCPUs |
| RAM | 12 GB |
| Disk type (SSD or hard disk) | Google Persistent Disk (SSD-backed, non-rotational) |
| Operating system | Ubuntu 24.04.4 LTS |
| Full run time, start to finish (50 files) | Not completed before the deadline. Last measurement: 82 of ~1,600 videos (711,578 frames) done after 3,849 s, at a steady 185 fps (model busy 96%, decode busy 90%). At that speed a complete run on this T4 would take about 16–21 h. |
| Command used | `python run.py --input finevideo/data --output results.jsonl` (INT8 TensorRT engine built by `python prepare.py`) |
