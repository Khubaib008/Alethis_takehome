# Hardware used for the full 50-file run

| Item | Value |
|---|---|
| Where it ran (own computer / university machine / Colab / Kaggle / rented GPU) | Google Colab (free tier) |
| GPU model and memory | NVIDIA Tesla T4, 15 GB (15,360 MiB) |
| CPU model and number of cores / threads | Intel Xeon @ 2.00 GHz, 2 vCPUs |
| RAM | 12 GB |
| Disk type (SSD or hard disk) | Google Persistent Disk (SSD-backed, non-rotational) |
| Operating system | Ubuntu 24.04.4 LTS |
| Full run time, start to finish (50 files) | Not completed before the deadline: stopped after TODO_SECONDS s with TODO_VIDEOS of ~1,600 videos done. Steady speed was 185 fps (model busy 96%, decode busy 90%), so a complete run on this T4 would take about 16–21 h. |
| Command used | `python run.py --input finevideo/data --output results.jsonl` (INT8 TensorRT engine built by `python prepare.py`) |
