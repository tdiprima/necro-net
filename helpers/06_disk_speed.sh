#!/bin/bash
# 🚀 Disk speed reality check (no guessing)

# sudo dnf install -y fio
fio --name=readtest --filename=/tmp/fio_test --size=1G --rw=read --bs=1M --numjobs=1 --time_based --runtime=30 --group_reporting

#| Storage  | Read speed         |
#| -------- | ------------------ |
#| HDD      | 100–200 MB/s 😬    |
#| SATA SSD | 400–550 MB/s 🙂    |
#| NVMe     | 1500–7000+ MB/s 🚀 |
