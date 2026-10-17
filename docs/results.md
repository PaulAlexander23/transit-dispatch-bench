# Results

<!-- results -->
Baselines over 512 episodes each (seed 0). Score: random = 0, the env's heuristic = 1. Return ± standard error.

| Env | Policy | Return | Score | Domain metrics |
|---|---|---|---|---|
| `bus/busy-v0` | random | -1563.2 ± 4.0 | 0.00 | mean wait 324 s, headway CV 0.81, bunching 0.139 |
| `bus/busy-v0` | always go | -1436.5 ± 3.8 | 0.79 | mean wait 309 s, headway CV 0.90, bunching 0.211 |
| `bus/busy-v0` | heuristic | -1403.3 ± 3.4 | 1.00 | mean wait 221 s, headway CV 0.47, bunching 0.000 |
| `bus/closure-v0` | random | -1747.1 ± 5.5 | 0.00 | mean wait 428 s, headway CV 0.80, bunching 0.115 |
| `bus/closure-v0` | always go | -1617.2 ± 6.1 | 0.85 | mean wait 368 s, headway CV 0.87, bunching 0.165 |
| `bus/closure-v0` | heuristic | -1595.0 ± 6.9 | 1.00 | mean wait 323 s, headway CV 0.52, bunching 0.002 |
| `bus/full-v0` | random | -1064.5 ± 1.8 | 0.00 | mean wait 258 s, headway CV 0.70, bunching 0.086 |
| `bus/full-v0` | always go | -983.5 ± 1.7 | 1.00 | mean wait 226 s, headway CV 0.71, bunching 0.113 |
| `bus/full-v0` | heuristic | -983.4 ± 1.7 | 1.00 | mean wait 180 s, headway CV 0.44, bunching 0.000 |
| `bus/loop-v0` | random | -633.6 ± 1.9 | 0.00 | mean wait 221 s, headway CV 0.77, bunching 0.135 |
| `bus/loop-v0` | always go | -559.3 ± 1.5 | 0.92 | mean wait 187 s, headway CV 0.69, bunching 0.124 |
| `bus/loop-v0` | heuristic | -552.7 ± 1.5 | 1.00 | mean wait 151 s, headway CV 0.43, bunching 0.000 |
| `bus/noisy-v0` | random | -668.1 ± 2.1 | 0.00 | mean wait 237 s, headway CV 0.79, bunching 0.129 |
| `bus/noisy-v0` | always go | -607.5 ± 2.0 | 0.78 | mean wait 223 s, headway CV 0.80, bunching 0.159 |
| `bus/noisy-v0` | heuristic | -590.2 ± 1.8 | 1.00 | mean wait 165 s, headway CV 0.46, bunching 0.000 |
| `bus/peak-v0` | random | -1712.0 ± 4.2 | 0.00 | mean wait 404 s, headway CV 0.76, bunching 0.101 |
| `bus/peak-v0` | always go | -1581.5 ± 4.4 | 0.76 | mean wait 347 s, headway CV 0.83, bunching 0.153 |
| `bus/peak-v0` | heuristic | -1539.6 ± 3.5 | 1.00 | mean wait 288 s, headway CV 0.46, bunching 0.000 |
| `elevator/full-v0` | random | 6.1 ± 0.1 | 0.00 | delivered/ep 6.1 |
| `elevator/full-v0` | heuristic | 67.1 ± 0.2 | 1.00 | delivered/ep 67.1 |
| `elevator/hall_calls-v0` | random | 5.7 ± 0.1 | 0.00 | delivered/ep 5.7 |
| `elevator/hall_calls-v0` | heuristic | 85.7 ± 0.3 | 1.00 | delivered/ep 85.7 |
| `elevator/kinematic-v0` | random | 4.5 ± 0.1 | 0.00 | delivered/ep 4.5 |
| `elevator/kinematic-v0` | heuristic | 60.4 ± 0.2 | 1.00 | delivered/ep 60.4 |
| `elevator/multi-v0` | random | 15.4 ± 0.2 | 0.00 | delivered/ep 15.4 |
| `elevator/multi-v0` | heuristic | 216.0 ± 0.3 | 1.00 | delivered/ep 216.0 |
| `elevator/original-v0` | random | 18.6 ± 0.3 | 0.00 | delivered/ep 18.6 |
| `elevator/original-v0` | heuristic | 136.4 ± 0.3 | 1.00 | delivered/ep 136.4 |
| `elevator/tall-v0` | random | 4.0 ± 0.1 | 0.00 | delivered/ep 4.0 |
| `elevator/tall-v0` | heuristic | 58.3 ± 0.2 | 1.00 | delivered/ep 58.3 |
| `train/blockage-v0` | random | -105.0 ± 4.1 | 0.00 | delay/train 13.13 min, CVaR10 35.80 min, deadlocks 0.525 |
| `train/blockage-v0` | first come first served | -68.2 ± 4.1 | 0.45 | delay/train 8.53 min, CVaR10 34.25 min, deadlocks 0.373 |
| `train/blockage-v0` | section reservation | -34.3 ± 2.9 | 0.87 | delay/train 4.29 min, CVaR10 26.66 min, deadlocks 0.109 |
| `train/blockage-v0` | heuristic | -23.7 ± 1.7 | 1.00 | delay/train 2.96 min, CVaR10 14.96 min, deadlocks 0.002 |
| `train/junction-v0` | random | -234.9 ± 2.1 | 0.00 | delay/train 29.37 min, CVaR10 36.30 min, deadlocks 0.990, diverted/ep 4.95 |
| `train/junction-v0` | first come first served | -68.2 ± 4.1 | 0.75 | delay/train 8.53 min, CVaR10 34.25 min, deadlocks 0.371, diverted/ep 0.00 |
| `train/junction-v0` | section reservation | -35.8 ± 3.0 | 0.90 | delay/train 4.48 min, CVaR10 27.61 min, deadlocks 0.000, diverted/ep 0.00 |
| `train/junction-v0` | route setting | -24.1 ± 1.8 | 0.95 | delay/train 3.01 min, CVaR10 15.38 min, deadlocks 0.000, diverted/ep 0.00 |
| `train/junction-v0` | heuristic | -14.0 ± 0.8 | 1.00 | delay/train 1.75 min, CVaR10 7.05 min, deadlocks 0.000, diverted/ep 1.46 |
| `train/line-v0` | random | -36.6 ± 0.5 | 0.00 | delay/train 3.05 min, CVaR10 5.20 min, deadlocks 0.000 |
| `train/line-v0` | first come first served | -10.9 ± 0.4 | 1.00 | delay/train 0.91 min, CVaR10 2.73 min, deadlocks 0.000 |
| `train/line-v0` | section reservation | -10.9 ± 0.4 | 1.00 | delay/train 0.91 min, CVaR10 2.73 min, deadlocks 0.000 |
| `train/line-v0` | heuristic | -10.9 ± 0.4 | 1.00 | delay/train 0.91 min, CVaR10 2.73 min, deadlocks 0.000 |
| `train/passing-v0` | random | -52.2 ± 0.5 | 0.00 | delay/train 4.35 min, CVaR10 6.45 min, deadlocks 0.000 |
| `train/passing-v0` | first come first served | -26.2 ± 0.4 | 0.88 | delay/train 2.19 min, CVaR10 4.00 min, deadlocks 0.000 |
| `train/passing-v0` | section reservation | -26.2 ± 0.4 | 0.88 | delay/train 2.19 min, CVaR10 4.00 min, deadlocks 0.000 |
| `train/passing-v0` | route setting | -26.2 ± 0.4 | 0.88 | delay/train 2.19 min, CVaR10 4.00 min, deadlocks 0.000 |
| `train/passing-v0` | heuristic | -22.8 ± 0.4 | 1.00 | delay/train 1.90 min, CVaR10 3.84 min, deadlocks 0.000 |
| `train/response-v0` | random | -166.9 ± 0.7 | 0.00 | cost/train 20.86 min, CVaR10 24.07 min, deadlocks 0.006, terminated/ep 7.73 |
| `train/response-v0` | first come first served | -76.8 ± 4.3 | 0.65 | cost/train 9.60 min, CVaR10 34.84 min, deadlocks 0.369, terminated/ep 0.00 |
| `train/response-v0` | section reservation | -46.5 ± 3.3 | 0.87 | cost/train 5.81 min, CVaR10 30.30 min, deadlocks 0.129, terminated/ep 0.00 |
| `train/response-v0` | route setting | -37.9 ± 2.7 | 0.93 | cost/train 4.74 min, CVaR10 24.99 min, deadlocks 0.006, terminated/ep 0.00 |
| `train/response-v0` | heuristic | -28.5 ± 1.7 | 1.00 | cost/train 3.56 min, CVaR10 16.07 min, deadlocks 0.000, terminated/ep 0.63 |
| `train/single-v0` | random | -80.7 ± 3.8 | 0.00 | delay/train 10.09 min, CVaR10 33.48 min, deadlocks 0.398 |
| `train/single-v0` | first come first served | -41.3 ± 3.4 | 0.56 | delay/train 5.16 min, CVaR10 30.34 min, deadlocks 0.225 |
| `train/single-v0` | section reservation | -13.6 ± 1.5 | 0.95 | delay/train 1.70 min, CVaR10 9.60 min, deadlocks 0.031 |
| `train/single-v0` | heuristic | -9.8 ± 0.5 | 1.00 | delay/train 1.22 min, CVaR10 4.60 min, deadlocks 0.000 |
<!-- /results -->
