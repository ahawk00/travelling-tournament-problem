# Travelling Tournament Problem (TTP)
This repository provides tools to generate, evaluate, and optimize tournament schedules under the Travelling Tournament Problem framework. It includes random schedule generation, Hill Climbing, Simulated Annealing, and repair mechanisms using both CP-SAT and deterministic methods.

## Random Schedule Generator 

Generates random double round-robin schedules for the TTP, evaluates distance and constraint violations.

Run: ```bash python random_search.py --n_schedules 5000```

Saved to results/random_search/{row,column}/:
- **`{n}.txt`**  
  Contains all generated schedules for `n` teams. Each row includes:
  - Number of teams (`n`)
  - Schedule type (`row` or `column`)
  - Total travel distance
  - Per-team travel distance
  - Constraint violations (total, double round-robin, no repeat, max streak)
  - Schedule matrix

- **`average.txt`**  
  Summarizes average results across all generated schedules for each team size:
  - Number of teams
  - Schedule type
  - Average total travel distance
  - Average per-team distance
  - Average constraint violations
  - Runtime (seconds)

- **`best.txt`**  
  The best (lowest-distance) schedule for each team size:
  - Number of teams
  - Schedule type
  - Travel distance (total and per team)
  - Constraint violations
  - Schedule matrix

- **`worst.txt`**  
  The worst (highest-distance) schedule for each team size, same format as `best.txt`
	
## Run Hill Climbing Instance(s)
Performs Hill Climbing to improve schedules based on distance or violations iteratively.

Run:
```bash python hill_climbing.py --instance SM4 --iterations 50000 --schedule_type row --objective distance --trials 3 ```

Results are saved to: results/HC_{iterations}/{objective}/{row,column}/{trial}/
Each trial produces:
- `best_{dataset}.txt`: Summary of the best result per trial:
  - Number of teams
  - Schedule type
  - Initial and final distances (total and per team)
  - Accepted mutations
  - Initial and final constraint violations
  - Final schedule
  - Runtime
- `{dataset}.txt`: Iteration-level log (every 10,000 iterations)
  - Tracks current distance, violations, schedule state, and mutation type


## Run Simulated Annealing Instance(s)
Runs Simulated Annealing with customisable cooling schedules to optimise schedules by reducing distance or violations.

Run:
```bash python simulated_annealing.py --instance SM12 --iterations 100000 --schedule_type row --objective distance --cooling geometric --trials 3 ```

Results are saved in: results/SA_{iterations}/{cooling}/{row,column}/{trial}/
Each trial produces:
- `best_{dataset}.txt`: Summary of the best result per trial:
  - Number of teams
  - Schedule type
  - Initial and final distances (total and per team)
  - Accepted mutations
  - Initial and final constraint violations
  - Final schedule
  - Runtime
- `{dataset}.txt`: Iteration-level log (every 10,000 iterations)
  - Tracks current distance, violations, schedule state, and mutation type

## Simulated Annealing with CP-SAT Repair
Runs SA and periodically repairs (every 10,000 iterations) schedules using the CP-SAT solver.

```bash python CP_SAT_constraint_repair.py --instance SM12 --iterations 100000 --schedule_type column --cooling geometric --trials 3 --cpsat_mode strict```

Results are saved in: results/{CP_SAT_Strict|CP_SAT_Soft}_{iterations}/{cooling_type}/{schedule_type}/{trial}/
- **`best_distance_{dataset}.txt`**: Summary of the best low-distance result per trial:
  - Number of teams
  - Schedule type
  - Initial and final total distances
  - Initial and final per-team distances
  - Accepted mutations
  - Initial and final constraint violations
  - Number of successful CP-SAT repairs
  - Average schedule difference after CP repair
  - Initial and final schedules
  - Runtime

- **`best_violations_{dataset}.txt`**: Summary of the best feasible (zero violation) result per trial:
  - Same columns as `best_distance_{dataset}.txt`, but final distance/violations are for the best constraint-feasible schedule

- **`{dataset}.txt`**: Iteration-level log (written every 10,000 iterations):
  - Iteration number
  - Number of teams
  - Schedule type
  - Current total and per-team distance
  - Current constraint violations
  - Current schedule (on milestone iterations)
  - CP-SAT repair status (e.g., "Feasible", "Not feasible")
  - Absolute difference from schedule before repair

## Simulated Annealing with Deterministic Repair
SA combined with rule-based (deterministic) constraint repair, using either row- or column-based schedule initialization. Repair every 10,000 iterations

### Row-first initialisation schedule repair
```bash python row_deterministic_repair.py --instance SM10 --iterations 100000  --cooling geometric --trials 5```

### Column-first initilisation schedule repair
```bash python col_deterministic_repair.py --instance SM10 --iterations 100000  --cooling geometric --trials 5```

Results are saved under: results/deterministic_repair_{iterations}/{schedule_type}/{trial}/
- **`best_{dataset}.txt`**: Summary of the best result per trial:
  - Number of teams
  - Schedule type
  - Initial and final total distances
  - Initial and final per-team distances
  - Accepted mutations
  - Initial and final constraint violations
  - Average schedule difference from repair steps
  - Initial and final schedules
  - Runtime

- **`{dataset}.txt`**: Iteration-level log (written every 10,000 iterations):
  - Iteration number
  - Number of teams
  - Schedule type
  - Current total and per-team distance
  - Current constraint violations
  - Current schedule (every 10k iterations)
  - Absolute schedule difference (on repair iterations)


