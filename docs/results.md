# Results

<!-- results -->
Baselines over 512 episodes each (seed 0). Score: random = 0, the env's heuristic = 1. Return ± standard error.

| Env | Policy | Return | Score | Domain metrics |
|---|---|---|---|---|
| `bus/busy-v0` | random | -1563.2 ± 4.0 | 0.00 | mean wait 324 s, headway CV 0.81, bunching 0.139 |
| `bus/busy-v0` | always go | -1436.5 ± 3.8 | 0.79 | mean wait 309 s, headway CV 0.90, bunching 0.211 |
| `bus/busy-v0` | heuristic | -1403.3 ± 3.4 | 1.00 | mean wait 221 s, headway CV 0.47, bunching 0.000 |
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
| `train/blockage-v0` | random | -108.1 ± 4.1 | 0.00 | delay/train 13.51 min, CVaR10 35.74 min, deadlocks 0.514 |
| `train/blockage-v0` | first come first served | -67.3 ± 4.0 | 0.48 | delay/train 8.42 min, CVaR10 33.81 min, deadlocks 0.361 |
| `train/blockage-v0` | section reservation | -29.5 ± 2.6 | 0.92 | delay/train 3.69 min, CVaR10 23.68 min, deadlocks 0.072 |
| `train/blockage-v0` | heuristic | -22.7 ± 1.7 | 1.00 | delay/train 2.84 min, CVaR10 14.71 min, deadlocks 0.000 |
| `train/line-v0` | random | -36.1 ± 0.4 | 0.00 | delay/train 3.01 min, CVaR10 4.80 min, deadlocks 0.000 |
| `train/line-v0` | first come first served | -10.4 ± 0.4 | 1.00 | delay/train 0.87 min, CVaR10 2.44 min, deadlocks 0.000 |
| `train/line-v0` | section reservation | -10.4 ± 0.4 | 1.00 | delay/train 0.87 min, CVaR10 2.44 min, deadlocks 0.000 |
| `train/line-v0` | heuristic | -10.4 ± 0.4 | 1.00 | delay/train 0.87 min, CVaR10 2.44 min, deadlocks 0.000 |
| `train/passing-v0` | random | -52.2 ± 0.4 | 0.00 | delay/train 4.35 min, CVaR10 6.12 min, deadlocks 0.000 |
| `train/passing-v0` | first come first served | -25.6 ± 0.4 | 1.00 | delay/train 2.14 min, CVaR10 3.67 min, deadlocks 0.000 |
| `train/passing-v0` | section reservation | -25.6 ± 0.4 | 1.00 | delay/train 2.14 min, CVaR10 3.67 min, deadlocks 0.000 |
| `train/passing-v0` | heuristic | -25.6 ± 0.4 | 1.00 | delay/train 2.14 min, CVaR10 3.67 min, deadlocks 0.000 |
| `train/response-v0` | random | -167.1 ± 0.7 | 0.00 | cost/train 20.89 min, CVaR10 24.14 min, deadlocks 0.008, terminated/ep 7.78 |
| `train/response-v0` | first come first served | -79.0 ± 4.2 | 0.63 | cost/train 9.87 min, CVaR10 34.61 min, deadlocks 0.389, terminated/ep 0.00 |
| `train/response-v0` | section reservation | -42.8 ± 3.2 | 0.89 | cost/train 5.35 min, CVaR10 29.24 min, deadlocks 0.105, terminated/ep 0.00 |
| `train/response-v0` | route setting | -36.6 ± 2.6 | 0.93 | cost/train 4.57 min, CVaR10 23.89 min, deadlocks 0.010, terminated/ep 0.00 |
| `train/response-v0` | heuristic | -27.5 ± 1.7 | 1.00 | cost/train 3.44 min, CVaR10 15.39 min, deadlocks 0.004, terminated/ep 0.66 |
| `train/single-v0` | random | -81.3 ± 3.8 | 0.00 | delay/train 10.17 min, CVaR10 34.39 min, deadlocks 0.387 |
| `train/single-v0` | first come first served | -41.0 ± 3.3 | 0.55 | delay/train 5.13 min, CVaR10 29.64 min, deadlocks 0.240 |
| `train/single-v0` | section reservation | -9.1 ± 0.7 | 0.99 | delay/train 1.13 min, CVaR10 4.36 min, deadlocks 0.006 |
| `train/single-v0` | heuristic | -8.6 ± 0.4 | 1.00 | delay/train 1.08 min, CVaR10 3.61 min, deadlocks 0.000 |
<!-- /results -->
