#!/bin/bash
# Are you on HDD or NVMe?
# ROTA = 1 → HDD (bad for ML)
# ROTA = 0 → SSD / NVMe (good)

lsblk -o NAME,ROTA,TYPE,SIZE,MOUNTPOINT
