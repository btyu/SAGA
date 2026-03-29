from typing import List, Callable, Optional

from scileo_agent.core.data_models import Population, Objective, Candidate
from modules.small_molecule_drug_design.utils.rdkit_utils import (
    calculate_tanimoto_similarity,
)


class SurvivalSelectionStrategyBase:

    def select(
        self,
        current_population: Population,
        new_population_with_scores: Population,
        objectives: List[Objective],
    ) -> Population:
        raise NotImplementedError


class FitnessSurvivalSelection(SurvivalSelectionStrategyBase):

    def __init__(
        self,
        *,
        population_size: int,
        elitism_fraction: float,
        survival_tanimoto_threshold: float,
        seed: int,
        init_group: str,
        remove_duplicates: Callable[[List[Candidate]], List[Candidate]],
        compute_candidate_score: Callable[[Candidate, List[Objective]], float],
        get_elite_candidates: Callable[
            [Population, List[Objective], int], List[Candidate]
        ],
        sample_smiles: Callable[[int, int, str], List[str]],
        sanitize_smiles_value: Callable[[str], str],
        elitism_fields: Optional[List[str]] = None,
    ) -> None:
        self.population_size = population_size
        self.elitism_fraction = elitism_fraction
        self.survival_tanimoto_threshold = survival_tanimoto_threshold
        self.seed = seed
        self.init_group = init_group
        self.remove_duplicates = remove_duplicates
        self.compute_candidate_score = compute_candidate_score
        self.get_elite_candidates = get_elite_candidates
        self.sample_smiles = sample_smiles
        self.sanitize_smiles_value = sanitize_smiles_value
        self.elitism_fields = elitism_fields or []

    def select(
        self,
        current_population: Population,
        new_population_with_scores: Population,
        objectives: List[Objective],
    ) -> Population:
        """
        Select survivors by combining parents and offspring, then selecting top N.

        Parents compete with offspring for survival (standard GA behavior).
        Elites from current population are guaranteed to survive.
        """
        elite_count = max(1, int(self.population_size * self.elitism_fraction))
        elite_count = min(elite_count, len(current_population.candidates))

        # Get elites (guaranteed to survive)
        elites = self.get_elite_candidates(current_population, objectives, elite_count)
        elite_repr = {c.representation for c in elites}

        # Combine parents + offspring, then deduplicate
        combined = current_population.candidates + new_population_with_scores.candidates
        combined_unique = self.remove_duplicates(combined)

        # Remove elites from competition pool (they're already selected)
        non_elite_pool = [
            c for c in combined_unique if c.representation not in elite_repr
        ]

        # Sort combined pool by fitness (parents compete with offspring)
        sorted_pool = sorted(
            non_elite_pool,
            key=lambda c: self.compute_candidate_score(c, objectives),
            reverse=True,
        )

        # Select top candidates to fill remaining slots
        remaining = max(0, self.population_size - len(elites))
        selected_candidates = elites + sorted_pool[:remaining]

        # Fallback if not enough candidates
        if len(selected_candidates) < self.population_size:
            needed = self.population_size - len(selected_candidates)
            fallback_smiles = self.sample_smiles(
                needed, seed=self.seed + 23, init_group=self.init_group
            )
            fallback_candidates = [
                Candidate(representation=self.sanitize_smiles_value(s))
                for s in fallback_smiles
            ]
            pool = self.remove_duplicates(selected_candidates + fallback_candidates)
            selected_candidates = pool[: self.population_size]

        return Population(candidates=selected_candidates)


class DiverseTopSurvivalSelection(SurvivalSelectionStrategyBase):

    def __init__(
        self,
        *,
        population_size: int,
        elitism_fraction: float,
        survival_tanimoto_threshold: float,
        seed: int,
        init_group: str,
        remove_duplicates: Callable[[List[Candidate]], List[Candidate]],
        compute_candidate_score: Callable[[Candidate, List[Objective]], float],
        get_elite_candidates: Callable[
            [Population, List[Objective], int], List[Candidate]
        ],
        sample_smiles: Callable[[int, int, str], List[str]],
        sanitize_smiles_value: Callable[[str], str],
        elitism_fields: Optional[List[str]] = None,
    ) -> None:
        self.population_size = population_size
        self.elitism_fraction = elitism_fraction
        self.survival_tanimoto_threshold = survival_tanimoto_threshold
        self.seed = seed
        self.init_group = init_group
        self.remove_duplicates = remove_duplicates
        self.compute_candidate_score = compute_candidate_score
        self.get_elite_candidates = get_elite_candidates
        self.sample_smiles = sample_smiles
        self.sanitize_smiles_value = sanitize_smiles_value
        self.elitism_fields = elitism_fields or []

        self.survival_leniency = 3

    def select(
        self,
        current_population: Population,
        new_population_with_scores: Population,
        objectives: List[Objective],
    ) -> Population:
        """
        Select survivors with diversity filtering.

        Parents compete with offspring for survival. Diversity is enforced
        by limiting similar molecules (based on Tanimoto threshold).
        """
        elite_count = max(1, int(self.population_size * self.elitism_fraction))
        elite_count = min(elite_count, len(current_population.candidates))

        # Get elites (guaranteed to survive)
        elites = self.get_elite_candidates(current_population, objectives, elite_count)
        elite_repr = {c.representation for c in elites}

        # Combine parents + offspring, then deduplicate
        combined = current_population.candidates + new_population_with_scores.candidates
        combined_unique = self.remove_duplicates(combined)

        # Remove elites from competition pool (they're already selected)
        non_elite_pool = [
            c for c in combined_unique if c.representation not in elite_repr
        ]

        # Sort combined pool by fitness (parents compete with offspring)
        sorted_pool = sorted(
            non_elite_pool,
            key=lambda c: self.compute_candidate_score(c, objectives),
            reverse=True,
        )

        # Select with diversity filtering
        selected = list(elites)
        threshold = float(self.survival_tanimoto_threshold)
        for cand in sorted_pool:
            if len(selected) >= self.population_size:
                break
            rep = cand.representation
            num_similar = 0
            for s in selected:
                try:
                    sim = calculate_tanimoto_similarity(rep, s.representation)
                except Exception:
                    sim = 1.0
                if sim >= threshold:
                    num_similar += 1

            if num_similar < self.survival_leniency:
                selected.append(cand)

        if len(selected) < self.population_size:
            needed = self.population_size - len(selected)
            fallback_smiles = self.sample_smiles(
                needed * 2, seed=self.seed + 17, init_group=self.init_group
            )
            for s in fallback_smiles:
                if len(selected) >= self.population_size:
                    break
                rep = self.sanitize_smiles_value(s)
                if rep in {c.representation for c in selected}:
                    continue
                is_diverse = True
                for c0 in selected:
                    try:
                        sim = calculate_tanimoto_similarity(rep, c0.representation)
                    except Exception:
                        sim = 1.0
                    if sim >= threshold:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append(Candidate(representation=rep))

        return Population(candidates=selected)


class ButinaClusterSurvivalSelection(SurvivalSelectionStrategyBase):

    def __init__(
        self,
        *,
        population_size: int,
        elitism_fraction: float,
        survival_tanimoto_threshold: float,
        seed: int,
        init_group: str,
        remove_duplicates: Callable[[List[Candidate]], List[Candidate]],
        compute_candidate_score: Callable[[Candidate, List[Objective]], float],
        get_elite_candidates: Callable[
            [Population, List[Objective], int], List[Candidate]
        ],
        sample_smiles: Callable[[int, int, str], List[str]],
        sanitize_smiles_value: Callable[[str], str],
        elitism_fields: Optional[List[str]] = None,
    ) -> None:
        self.population_size = population_size
        self.elitism_fraction = elitism_fraction
        self.survival_tanimoto_threshold = survival_tanimoto_threshold
        self.seed = seed
        self.init_group = init_group
        self.remove_duplicates = remove_duplicates
        self.compute_candidate_score = compute_candidate_score
        self.get_elite_candidates = get_elite_candidates
        self.sample_smiles = sample_smiles
        self.sanitize_smiles_value = sanitize_smiles_value
        self.elitism_fields = elitism_fields or []

    def select(
        self,
        current_population: Population,
        new_population_with_scores: Population,
        objectives: List[Objective],
    ) -> Population:
        """
        Select survivors using Butina clustering for diversity.

        Parents compete with offspring for survival. Clustering ensures
        diversity across the selected population.
        """
        from rdkit.ML.Cluster import Butina

        elite_count = max(1, int(self.population_size * self.elitism_fraction))
        elite_count = min(elite_count, len(current_population.candidates))

        elites = self.get_elite_candidates(
            current_population,
            objectives,
            elite_count,
            getattr(self, "elitism_fields", None),
        )
        elite_repr = {c.representation for c in elites}

        # Combine parents + offspring, then deduplicate
        combined = current_population.candidates + new_population_with_scores.candidates
        combined_unique = self.remove_duplicates(combined)

        # Remove elites from clustering pool (they're already selected)
        non_elite_pool = [
            c for c in combined_unique if c.representation not in elite_repr
        ]

        selected: List[Candidate] = list(elites)

        if non_elite_pool:
            # Precompute scores for ranking within clusters
            scores = {
                id(c): self.compute_candidate_score(c, objectives) for c in non_elite_pool
            }

            # Tanimoto distance based on existing similarity util
            def tanimoto_distance(rep_i: str, rep_j: str) -> float:
                try:
                    sim = calculate_tanimoto_similarity(rep_i, rep_j)
                except Exception:
                    sim = 0.0
                return 1.0 - float(sim)

            reps = [c.representation for c in non_elite_pool]
            dist_thresh = 1.0 - float(self.survival_tanimoto_threshold)
            clusters = Butina.ClusterData(
                reps,
                nPts=len(reps),
                distThresh=dist_thresh,
                isDistData=False,
                distFunc=tanimoto_distance,
                reordering=False,
            )
            # Minimal logging: report number of clusters formed
            try:
                print(
                    f"Butina clustering: {len(clusters)} clusters from {len(reps)} candidates"
                )
            except Exception:
                pass

            # Build per-cluster candidate lists sorted by score desc
            cluster_lists: List[List[Candidate]] = []
            for cl in clusters:
                members = [non_elite_pool[i] for i in cl]
                members_sorted = sorted(
                    members, key=lambda c: scores[id(c)], reverse=True
                )
                cluster_lists.append(members_sorted)

            # First pass: take top 1 from each cluster
            for members in cluster_lists:
                if len(selected) >= self.population_size:
                    break
                if members:
                    selected.append(members[0])

            # Round-robin fill from clusters until population_size
            # limit up to 3 per cluster
            MAX_PER_CLUSTER = 3  # limit to at most 3 selections per cluster
            per_cluster_counts = [min(1, len(m)) for m in cluster_lists]
            round_index = 1
            while len(selected) < self.population_size:
                made_progress = False
                for idx, members in enumerate(cluster_lists):
                    if len(selected) >= self.population_size:
                        break
                    if per_cluster_counts[idx] >= MAX_PER_CLUSTER:
                        continue
                    if round_index < len(members):
                        selected.append(members[round_index])
                        per_cluster_counts[idx] += 1
                        made_progress = True
                if not made_progress:
                    break
                round_index += 1

        # Fallback: keep same behavior as DiverseTopSurvivalSelection
        if len(selected) < self.population_size:
            threshold = float(self.survival_tanimoto_threshold)
            needed = self.population_size - len(selected)
            fallback_smiles = self.sample_smiles(
                needed * 2, seed=self.seed + 17, init_group=self.init_group
            )
            for s in fallback_smiles:
                if len(selected) >= self.population_size:
                    break
                rep = self.sanitize_smiles_value(s)
                if rep in {c.representation for c in selected}:
                    continue
                is_diverse = True
                for c0 in selected:
                    try:
                        sim = calculate_tanimoto_similarity(rep, c0.representation)
                    except Exception:
                        sim = 1.0
                    if sim >= threshold:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append(Candidate(representation=rep))

        return Population(candidates=selected)
