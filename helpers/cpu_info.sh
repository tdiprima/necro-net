#!/bin/bash
# 🧵 How many CPU cores can feed your GPU?

lscpu

# num_workers = <how many you can realistically run>
# num_workers ≈ min(8, number_of_CPU_threads / 2)
