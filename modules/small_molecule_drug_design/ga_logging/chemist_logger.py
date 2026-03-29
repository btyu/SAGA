from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import json
import random

# RDKit-based utility already provided in the project
from modules.small_molecule_drug_design.utils.rdkit_utils import (
    calculate_tanimoto_similarity, )


@dataclass
class LLMExample:
    kind: str  # "weights" | "crossover" | "mutation"
    prompt: str
    response: str
    smiles: Optional[str] = None


@dataclass
class GenerationRecord:
    generation: int
    examples: Dict[str, List[LLMExample]] = field(default_factory=lambda: {
        "crossover": [],
        "mutation": []
    })
    stages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    diversity: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)


class ChemistLogger:
    """
    Human-friendly logger for chemists.

    Produces:
    - Stage CSVs per generation (original, crossover, mutation, selected)
    - A Markdown report with sampled LLM prompts/responses and diversity
    - A JSON index for programmatic access
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "logs",
        max_examples_per_generation: int = 3,
        random_seed: int = 42,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_examples = int(max_examples_per_generation)
        self.random = random.Random(random_seed)

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Per-run subfolder for per-generation CSVs and human report
        self.run_dir = self.output_dir / "per_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_meta: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "created_at": self.created_at,
        }

        self.objectives_meta: Dict[str, Any] = {}
        self.weights_prompt: Optional[str] = None
        self.weights_response: Optional[str] = None
        self.final_weights: Optional[List[float]] = None

        self.generations: Dict[int, GenerationRecord] = {}

    # --- One-time metadata ---
    def set_objectives_info(self,
                            objectives: List[Any],
                            weights: Optional[List[float]] = None) -> None:
        self.objectives_meta = {
            "objectives": [{
                "name":
                getattr(o, "name", "unknown"),
                "direction":
                getattr(o, "optimization_direction", ""),
                "description":
                getattr(o, "description", ""),
            } for o in objectives]
        }
        if weights is not None:
            self.final_weights = list(weights)

    def set_run_context(self, ctx: Dict[str, Any]) -> None:
        """Attach run-level metadata (e.g., config) for persistence in outputs."""
        if isinstance(ctx, dict):
            # Ensure keys are strings for JSON serialization
            self.run_meta.update({str(k): v for k, v in ctx.items()})

    def log_weight_prompt(self, prompt: str, response: str,
                          weights: List[float]) -> None:
        self.weights_prompt = prompt
        self.weights_response = response
        self.final_weights = list(weights)

    # --- Per-generation content ---
    def log_llm_examples(self, generation: int, kind: str,
                         examples: List[LLMExample]) -> None:
        rec = self.generations.setdefault(generation,
                                          GenerationRecord(generation))
        buf = rec.examples.setdefault(kind, [])
        # Cap total examples per generation per kind
        remaining = max(0, self.max_examples - len(buf))
        if remaining <= 0:
            return
        buf.extend(examples[:remaining])

    def log_population_stage(
        self,
        generation: int,
        stage: str,
        candidates: List[Any],
        objectives: List[Any],
        agg_key: str = "multiobj_score",
    ) -> None:
        rec = self.generations.setdefault(generation,
                                          GenerationRecord(generation))
        rows: List[Dict[str, Any]] = []
        obj_names = [getattr(o, "name", "obj") for o in objectives]
        for c in candidates:
            data: Dict[str, Any] = {
                "smiles": getattr(c, "representation", None),
                "aggregate": getattr(c, "scores", {}).get(agg_key, None),
            }
            for name in obj_names:
                data[name] = getattr(c, "scores", {}).get(name, None)
            rows.append(data)

        rec.stages[stage] = rows

        # Persist table as CSV
        csv_path = (self.run_dir /
                    f"{self.experiment_name}_gen{generation:04d}_{stage}.csv")

        # Pretty-format numeric values (rounded) when writing
        def _format_cell(key: str, value: Any) -> Any:
            if key == "smiles":
                return value
            if value is None:
                return ""
            if isinstance(value, (int, )):
                return str(value)
            if isinstance(value, float):
                # NaN check: NaN != NaN
                if value != value:
                    return ""
                return f"{value:.4f}"
            return value

        formatted_rows = [{
            k: _format_cell(k, v)
            for k, v in row.items()
        } for row in rows]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["smiles", "aggregate", *obj_names])
            writer.writeheader()
            writer.writerows(formatted_rows)

        # Overwrite human report once per generation after 'selected' stage
        if stage == "selected":
            try:
                self.save_human_report()
            except Exception:
                # Non-fatal: continue even if report generation fails
                pass

    def log_diversity(self,
                      generation: int,
                      smiles_list: List[str],
                      sample_pairs: int = 1000) -> None:
        rec = self.generations.setdefault(generation,
                                          GenerationRecord(generation))
        n = len(smiles_list)
        if n < 2:
            rec.diversity = {
                "n": n,
                "mean_tanimoto": None,
                "internal_diversity": None
            }
            return

        max_pairs = n * (n - 1) // 2
        target = min(int(sample_pairs), max_pairs)
        idx_pairs: set[tuple[int, int]] = set()
        while len(idx_pairs) < target:
            i = self.random.randrange(n)
            j = self.random.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            idx_pairs.add((i, j))

        sims: List[float] = []
        for i, j in idx_pairs:
            s1, s2 = smiles_list[i], smiles_list[j]
            try:
                sims.append(calculate_tanimoto_similarity(s1, s2))
            except Exception:
                # Skip pairs that fail to parse
                continue

        mean_sim = sum(sims) / len(sims) if sims else None
        rec.diversity = {
            "n": n,
            "pairs_sampled": len(sims),
            "mean_tanimoto": mean_sim,
            "internal_diversity":
            (1 - mean_sim) if mean_sim is not None else None,
        }

    def log_timing(self, generation: int, timing_stats: Dict[str, Any]) -> None:
        """Log timing statistics for a generation.
        
        Args:
            generation: Generation number
            timing_stats: Dictionary with timing stats, e.g.:
                {
                    "llm_crossover": {"count": 10, "total_time": 5.2, "avg_time": 0.52},
                    "llm_mutation": {"count": 7, "total_time": 3.1, "avg_time": 0.44},
                    "scoring": {"count": 120, "total_time": 12.5, "avg_time": 0.104}
                }
        """
        rec = self.generations.setdefault(generation, GenerationRecord(generation))
        rec.timing = timing_stats

    # --- Finalization ---
    def save_human_report(self) -> Dict[str, str]:
        # Helper to compute aggregates (top1, top10_mean) over rows for given keys
        def _compute_aggregates(
                rows: List[Dict[str, Any]],
                keys: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
            aggregates: Dict[str, Dict[str, Optional[float]]] = {}
            if not rows:
                return aggregates
            for key in keys:
                values = [
                    v for v in (r.get(key) for r in rows)
                    if isinstance(v, (int, float))
                ]
                if not values:
                    aggregates[key] = {"top1": None, "top10_mean": None}
                    continue
                values_sorted = sorted(values, reverse=True)
                top1 = values_sorted[0]
                top10 = values_sorted[:min(10, len(values_sorted))]
                top10_mean = sum(top10) / len(top10) if top10 else None
                aggregates[key] = {
                    "top1":
                    float(top1),
                    "top10_mean":
                    float(top10_mean) if top10_mean is not None else None
                }
            return aggregates

        # JSON index
        index = {
            "meta": self.run_meta,
            "objectives": self.objectives_meta,
            "weights": {
                "prompt": self.weights_prompt,
                "response": self.weights_response,
                "final_weights": self.final_weights,
            },
            "generations": {
                gen: {
                    "diversity":
                    rec.diversity,
                    "timing": rec.timing,
                    "examples": {
                        k: [ex.__dict__ for ex in v]
                        for k, v in rec.examples.items()
                    },
                    "stages": {
                        k: len(v)
                        for k, v in rec.stages.items()
                    },
                    # Aggregates for the 'selected' stage (if available)
                    "selected_aggregates":
                    ((lambda rows, keys: _compute_aggregates(rows, keys))(
                        rec.stages.get("selected", []), [
                            "aggregate", *[
                                o.get("name", "obj")
                                for o in self.objectives_meta.get(
                                    "objectives", [])
                            ]
                        ]) if rec.stages.get("selected") else {}),
                }
                for gen, rec in sorted(self.generations.items())
            },
        }

        idx_path = (self.run_dir / f"{self.experiment_name}_human_index.json")
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

        # Markdown report
        md_path = (self.run_dir / f"{self.experiment_name}_human_report.md")
        with open(md_path, "w") as f:
            f.write(
                f"# {self.experiment_name} (created {self.created_at})\n\n")

            # Objectives
            if self.objectives_meta:
                f.write("## Objectives\n")
                for o in self.objectives_meta.get("objectives", []):
                    name = o.get("name", "obj")
                    direction = o.get("direction", "")
                    desc = o.get("description", "")
                    f.write(f"- {name} ({direction}): {desc}\n")
                f.write("\n")

            # Weights
            if self.final_weights is not None:
                f.write("## Multi-objective weights\n")
                weights_str = ", ".join(
                    f"{w:.4f}" if isinstance(w, (int, float)) else str(w)
                    for w in self.final_weights)
                f.write(f"Weights: {weights_str}\n\n")
            if self.weights_prompt:
                f.write("### Weights prompt\n\n`````\n")
                f.write(self.weights_prompt.strip() + "\n")
                f.write("`````\n\n")
            if self.weights_response:
                f.write("### Weights response\n\n`````\n")
                f.write(self.weights_response.strip() + "\n")
                f.write("`````\n\n")

            # Generations
            for gen, rec in sorted(self.generations.items()):
                f.write(f"## Generation {gen}\n")
                if rec.diversity:
                    f.write("- Diversity: " + json.dumps(rec.diversity) + "\n")
                if rec.timing:
                    f.write("- Timing:\n")
                    # Group scoring objectives together
                    scoring_ops = {}
                    other_ops = {}
                    for op_name, op_stats in rec.timing.items():
                        if isinstance(op_stats, dict) and op_stats.get("count", 0) > 0:
                            if op_name.startswith("scoring_"):
                                scoring_ops[op_name] = op_stats
                            elif op_name == "scoring":
                                scoring_ops[op_name] = op_stats
                            else:
                                other_ops[op_name] = op_stats
                    
                    # Print LLM operations first
                    for op_name, op_stats in sorted(other_ops.items()):
                        count = op_stats.get("count", 0)
                        total_time = op_stats.get("total_time", 0.0)
                        avg_time = op_stats.get("avg_time", 0.0)
                        if "llm" in op_name.lower():
                            f.write(f"  - {op_name}: {count} calls, {total_time:.2f}s total (avg: {avg_time:.3f}s/call)\n")
                        else:
                            f.write(f"  - {op_name}: {count} operations, {total_time:.2f}s total (avg: {avg_time:.3f}s/op)\n")
                    
                    # Print scoring operations
                    if scoring_ops:
                        # Print total scoring first
                        if "scoring" in scoring_ops:
                            op_stats = scoring_ops["scoring"]
                            count = op_stats.get("count", 0)
                            total_time = op_stats.get("total_time", 0.0)
                            avg_time = op_stats.get("avg_time", 0.0)
                            f.write(f"  - Scoring (all objectives): {count} compounds, {total_time:.2f}s total (avg: {avg_time:.4f}s/compound)\n")
                        
                        # Print per-objective scoring
                        per_obj_ops = {k: v for k, v in scoring_ops.items() if k != "scoring"}
                        if per_obj_ops:
                            f.write(f"  - Scoring per objective:\n")
                            for op_name in sorted(per_obj_ops.keys()):
                                op_stats = per_obj_ops[op_name]
                                obj_name = op_name.replace("scoring_", "")
                                count = op_stats.get("count", 0)
                                total_time = op_stats.get("total_time", 0.0)
                                avg_time = op_stats.get("avg_time", 0.0)
                                f.write(f"    - {obj_name}: {count} compounds, {total_time:.2f}s total (avg: {avg_time:.4f}s/compound)\n")
                # Aggregated metrics from selected stage (if present)
                if "selected" in rec.stages:
                    rows = rec.stages["selected"]
                    keys = [
                        "aggregate", *[
                            o.get("name", "obj")
                            for o in self.objectives_meta.get(
                                "objectives", [])
                        ]
                    ]
                    aggs = _compute_aggregates(rows, keys)
                    if aggs:
                        f.write("- Aggregates (selected):\n")
                        for k, v in aggs.items():
                            top1 = v.get("top1")
                            top10m = v.get("top10_mean")
                            top1_str = f"{top1:.4f}" if isinstance(
                                top1, (int, float)) else "NA"
                            top10_str = f"{top10m:.4f}" if isinstance(
                                top10m, (int, float)) else "NA"
                            f.write(
                                f"  - {k}: top1={top1_str}, top10_mean={top10_str}\n"
                            )
                # Stage CSV references in logical order
                for stage in ["original", "crossover", "mutation", "selected"]:
                    if stage in rec.stages:
                        csv_name = (
                            f"{self.experiment_name}_gen{gen:04d}_{stage}.csv")
                        f.write(f"- {stage.capitalize()} table: {csv_name}\n")
                # Examples
                for kind in ["crossover", "mutation"]:
                    examples = rec.examples.get(kind, [])
                    if not examples:
                        continue
                    f.write(f"\n### {kind.title()} examples\n")
                    for i, ex in enumerate(examples, 1):
                        f.write(f"\n#### Example {i}\n")
                        f.write("Prompt:\n`````\n")
                        f.write(ex.prompt + "\n")
                        f.write("`````\n")
                        f.write("Response:\n`````\n")
                        f.write(ex.response + "\n")
                        f.write("`````\n")
                        if ex.smiles:
                            f.write(f"Offspring SMILES: `{ex.smiles}`\n")
                    f.write("\n")
                f.write("\n")

        return {"index": str(idx_path), "report": str(md_path)}
