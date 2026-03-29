# GA Logger for Molecular Optimization

Tracks genetic algorithm optimization progress with comprehensive statistics and visualization.

## Features

- **Population tracking**: Store top candidates per generation
- **Statistics**: Mean, median, top-1, top-10 average scores
- **Visualization**: Convergence plots and distribution evolution
- **Data export**: CSV, pickle, and JSON formats
- **Memory efficient**: Stores only top-K candidates

## Usage

```python
from modules.small_molecule_drug_design.ga_logging import GALogger

# Create logger
logger = GALogger(
    objectives=objectives,
    store_top_k=100,
    experiment_name="my_experiment"
)

# Use with optimizer
result = optimizer.optimize(
    current_population=None,
    objectives=objectives,
    logger=logger
)

# Generate report and plots
report = logger.get_summary_report()
logger.plot_convergence(save_path="convergence.png")
```

## Key Methods

- `log_generation()`: Log population statistics for a generation
- `get_summary_report()`: Generate comprehensive summary
- `plot_convergence()`: Create convergence plots
- `get_best_candidates()`: Retrieve top candidates across all generations
- `save_log()`: Export data to files

## Output Files

- `*_convergence.csv`: Generation-by-generation statistics
- `*_populations.pkl`: Candidate populations
- `*_summary.json`: Optimization summary report 