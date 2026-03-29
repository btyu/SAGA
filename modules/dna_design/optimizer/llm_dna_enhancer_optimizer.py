from typing import List, Optional, Dict, Any
import re
import json

import numpy as np

from scileo_agent.utils.logging import avoid_logging_change, get_logger
with avoid_logging_change():
    import grelu
    import grelu.sequence.utils

from scileo_agent.core.registry import register_module, get_scorer
from scileo_agent.core.modules.optimizer import OptimizerModule
from scileo_agent.core.data_models import Population, Objective, Candidate, ObjectiveIndex
from scileo_agent.core.config import LLMConfig

logger = get_logger()

# === Prompt Templates ===
CELL_TYPE_PROMPT_TEMPLATE = """You will be provided with a high-level goal, and your task is to extract the cell type that should be optimized for from it. 

Here is the high-level goal:
{high_level_goal}

Your response should include a valid JSON list of the to-be-optimized cell type names, wrapped in <answer> ... </answer> tags. For example:
<answer>
["HepG2"]
</answer>
"""

DNA_OPTIMIZATION_PROMPT_TEMPLATE = """You are an expert in DNA enhancer and promoter design. No need to consider multi species or fold change. You are given {num_seqs} DNA sequences, each has {seq_length} base pairs in length, along with their associated property values in three cell lines: HepG2, K562, and SKNSH. Your task is to mutate each DNA sequence, using only the nucleotides A, T, G, C, and inert {cell_line}-specific motifs as DNA sequences, to make them as {cell_line}-specific enhancers with stability and diversity. {specific_task_instruction} Your response should include {num_seqs} DNA sequences, each exactly {seq_length} base pairs long (no need for optimization), and wrapped in <DNA> ... </DNA> tags.

Here are the inputs, where each line includes a DNA sequence and its MPRA expression of each cell line with other properties in the format: <DNA>DNA sequences</DNA>  [{regular_objective_names_str}].
{inputs_str}

{population_wise_objective_str}

Your response:"""

DNA_OPTIMIZATION_PROMPT_NODIV_TEMPLATE = """You are an expert in DNA enhancer and promoter design. No need to consider multi species or fold change. You are given {num_seqs} DNA sequences, each has {seq_length} base pairs in length, along with their associated property values in three cell lines: HepG2, K562, and SKNSH. Your task is to mutate each DNA sequence, using only the nucleotides A, T, G, C, and inert {cell_line}-specific motifs as DNA sequences, to make them as {cell_line}-specific enhancers. {specific_task_instruction} Your response should include {num_seqs} DNA sequences, each exactly {seq_length} base pairs long (no need for optimization), and wrapped in <DNA> ... </DNA> tags.

Here are the inputs, where each line includes a DNA sequence and its MPRA expression of each cell line with other properties in the format: <DNA>DNA sequences</DNA>  [{regular_objective_names_str}].
{inputs_str}

{population_wise_objective_str}

Your response:"""

# To Tianyu: the sentence `No need to consider multi species or fold change.` in the prompt templates is quite weird. Please consider describing it in a more natural way at a better place.

def is_valid_dna_sequence(sequence: str, seq_length: Optional[int] = None) -> bool:
    """Check if a sequence is a valid DNA sequence."""
    if not sequence or not isinstance(sequence, str):
        return False
    
    if seq_length is not None and len(sequence) != seq_length:
        return False
    
    # Check if all characters are valid DNA bases
    valid_bases = {'A', 'T', 'G', 'C'}
    sequence_upper = sequence.upper()
    
    if len(set(sequence_upper) - set(valid_bases)) != 0:
        return False

    return True


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return 0
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


# === Optimizer Implementation ===
@register_module("llm_dna_enhancer_optimizer", "1.0.0")
class LlmDnaEnhancerOptimizer(OptimizerModule):
    """
    Optimizer that uses an LLM to iteratively mutate and improve DNA enhancer sequences.
    """
    # The objectives that this optimizer requires, and we know they are regular objectives
    REQUIRED_OBJECTIVE_NAMES = ["dna_hepg2_enhancer_MPRA_expression", "dna_k562_enhancer_MPRA_expression", "dna_sknsh_enhancer_MPRA_expression", "dna_motif_num", 'dna_motif_num_human', "dna_hepg2_tissue_cage_expression", "dna_k562_tissue_cage_expression", "dna_sknsh_tissue_cage_expression","dna_diversity", "dna_stability"]

    # If the optimizer does not require initial population (you create it or your algorithm does not need it),
    # you can override this property to return False
    @property
    def requires_initial_population(self) -> bool:
        return False
    
    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        super().__init__(module_id=module_id, config=config, llm_config=llm_config)

        # This objective requires a LLM, so we want to ensure that it is available
        if not self.has_llm():
            raise RuntimeError("LLM client is not available. Please provide llm_config when initializing the module.")

        # Set defaults and allow override via config
        self.n_rounds = self.config.get("n_rounds", 5)
        self.batch_size = self.config.get("batch_size", 20)
        self.seq_length = self.config.get("seq_length", 200)
        self.num_initial_seqs = self.config.get("num_initial_seqs", 1000)
        self.cell_type_extraction_llm_model_name = "openai/gpt-4.1-nano-2025-04-14"
        self.use_diversity_for_filtering = self.config.get("use_diversity_for_filtering", True)
        self.diversity_filtering_threshold = self.config.get("diversity_filtering_threshold", 0.5)

        # Store high-level goals and corresponding cell lines to maximize
        self._goals_to_cell_lines = dict()

        # Maintain a set of objectives for evaluating candidates for filtering
        hepg2_scorer = get_scorer("dna_hepg2_enhancer_MPRA_expression")
        k562_scorer = get_scorer("dna_k562_enhancer_MPRA_expression")
        sknsh_scorer = get_scorer("dna_sknsh_enhancer_MPRA_expression")
        assert hepg2_scorer is not None, "HepG2 scorer is not found"
        assert k562_scorer is not None, "K562 scorer is not found"
        assert sknsh_scorer is not None, "SKNSH scorer is not found"
        self._filtering_objectives = {
            "HepG2": Objective(name="dna_hepg2_enhancer_MPRA_expression", description="", population_wise=False, scorer=hepg2_scorer),
            "K562": Objective(name="dna_k562_enhancer_MPRA_expression", description="", population_wise=False, scorer=k562_scorer),
            "SKNSH": Objective(name="dna_sknsh_enhancer_MPRA_expression", description="", population_wise=False, scorer=sknsh_scorer),
        }

    async def _create_initial_population(self) -> Population:
        # Generate random sequences
        candidates = await self.create_random_candidates(num_candidates=self.num_initial_seqs)
        return Population(candidates=candidates)

    async def create_random_candidates(self, num_candidates: int, **additional_kwargs: Dict[str, Any]) -> List[Candidate]:
        start_seq = grelu.sequence.utils.generate_random_sequences(
            n=num_candidates,
            seq_len=self.seq_length,
            seed=0,
            output_format="strings"
        )
        return [Candidate(representation=seq) for seq in start_seq]
    
    async def _extract_cell_types(self, high_level_goal: str) -> List[str]:
        prompt = CELL_TYPE_PROMPT_TEMPLATE.format(high_level_goal=high_level_goal)
        count = 0
        try:
            while True:
                response = (await self.call_llm_with_prompt_async(prompt))['content']
                # Extract the cell types from the response
                match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if match is None:
                    raise ValueError("No cell types found in the response")
                cell_types = json.loads(match.group(1))
                cell_types_new = []
                for item in cell_types:
                    if "SK-N-SH" in item:
                        cell_types_new.append("SKNSH")
                    else:
                        cell_types_new.append(item)
                cell_types = cell_types_new
                if len(cell_types) > 0:
                    return cell_types
                count += 1
                if count > 3:
                    raise ValueError("No cell types found in the response")
        except Exception as e:
            raise ValueError(f"Error extracting cell types: {e}")

    # This is the main function that runs the optimization
    async def optimize(
        self,
        current_population: Optional[Population],
        objectives: List[Objective],
        **additional_info: Dict[str, Any]
    ) -> Population:
        """
        Conduct optimization on the given population, towards the given objectives.

        Args:
            current_population: Current population of candidates, if None, this optimizer will need to create an initial population, usually in the first iteration
            objectives: List of objectives with scorer functions
            **additional_info: Additional information to pass to the optimizer
            
        Returns:
            Population of optimized candidates with their scores
        """

        # Get the high-level goal from the additional info
        high_level_goal = additional_info.get("high_level_goal", None)
        if high_level_goal is None:
            raise ValueError("High-level goal is not provided")
        
        # Get the cell types to optimize for from the high_level_goal
        if high_level_goal in self._goals_to_cell_lines:
            cell_lines_to_optimize = self._goals_to_cell_lines[high_level_goal]
        else:
            cell_lines_to_optimize = await self._extract_cell_types(high_level_goal)
            self._goals_to_cell_lines[high_level_goal] = cell_lines_to_optimize

        # This optimizer needs to have an initial population, so we need to create one if not provided
        if current_population is None or current_population.is_empty:
            if self.requires_initial_population:
                raise ValueError(f"{self.module_id} requires a non-empty initial population.")
            else:
                current_population = await self._create_initial_population()
        
        # Filter the initial population based on the required objectives
        while True:
            # Filter the initial population based on the required objectives
            current_population = await self._filter_candidates(current_population, cell_lines_to_optimize)

            if not current_population.is_empty:
                # If the initial population after filtering is not empty, break
                break
            else:
                # If the initial population is empty, it means that the initial population is not good
                # We need to create a new one
                current_population = await self._create_initial_population()

        final_candidates = []

        population = current_population

        # To Tianyu: Since we have already filtered the initial population, we don't need to filter it again?
        # population = await self._filter_candidates(population, cell_lines)

        logger.debug(f"Candidates to optimize after filtering: {population.size}")
        
        prepare_trying_times = self.n_rounds
        while len(final_candidates)==0 and prepare_trying_times>=0:
            for batch_idx in range(0, population.size, self.batch_size):  # Optimize batch_size sequences at a time
                # Select the batch
                begin = batch_idx
                end = min(batch_idx + self.batch_size, population.size)
                batch_candidates = population.candidates[begin:end]
                batch_population = Population(candidates=batch_candidates)

                # Optimize the batch
                for _ in range(self.n_rounds):
                    optimized_batch_population = await self._optimize_batch(batch_population, objectives, cell_lines_to_optimize)
                    # logger.debug(f"Batch {begin}-{end}, Round {_}: Optimized {optimized_batch_population.size} candidates")
                    if optimized_batch_population.is_empty:
                        break
                    optimized_batch_population = await self._filter_candidates(optimized_batch_population, cell_lines_to_optimize)
                    # logger.debug(f"Batch {begin}-{end}, Round {_}: Filtered {optimized_batch_population.size} candidates")
                    if optimized_batch_population.is_empty:
                        break
                    logger.debug(f"Batch {begin}-{end}, Round {_}: Got {optimized_batch_population.size} candidates")

                    final_candidates.extend(optimized_batch_population.candidates)

            if len(final_candidates) ==0:
                print("detected problem population")
                prepare_trying_times -= 1
        # Final evaluation
        
        if len(final_candidates) !=0:
            final_population = Population(candidates=final_candidates)
        else:
            final_population = await self._filter_candidates(population, cell_lines_to_optimize)
            logger.debug(f"Reach empty output, so we only use filter.")
        logger.debug(f"Final population size: {final_population.size}")

        return final_population

    # async def evaluate_population(self, population: Population, objectives: List[Objective], force_evaluation: bool = False) -> Population:
    #     """
    #     Evaluate candidates based on objectives.
        
    #     Args:
    #         population: Population of candidates to evaluate
    #         objectives: List of objectives with scorer functions
    #         force_evaluation: Whether to force evaluation of candidates, otherwise skip already evaluated candidates/population

    #     Returns:
    #         Population of candidates with their scores
    #     """
    #     logger.debug(f'Evaluating population of size {population.size} with objectives: {[obj.name for obj in objectives]}')
    #     return await super().evaluate_population(population, objectives, force_evaluation)

    async def _filter_candidates(self, population: Population, cell_lines_to_optimize: List[str]) -> Population:
        # Filter candidates based on the three basic objectives and optionally diversity
        # Specifically, only keep the candidates that have the maximum score for the to-be-optimized cell line
        # and the sequences are valid and optionally diverse

        if len(cell_lines_to_optimize) == 0:
            raise ValueError("No cell lines to optimize for")
        if len(cell_lines_to_optimize) > 1:
            raise NotImplementedError("More than one cell line to optimize for, which is not supported yet")
        cell_line = cell_lines_to_optimize[0]

        # Get the filtering objectives
        filtering_cell_lines = ['HepG2', 'K562', 'SKNSH']
        filtering_objectives = [self._filtering_objectives[cell_line_test] for cell_line_test in filtering_cell_lines]
        try:
            max_objective_idx = [item.lower() for item in filtering_cell_lines].index(cell_line.lower())
        except ValueError:
            raise ValueError(f"The cell line {cell_line} is not in the filtering cell lines {filtering_cell_lines}")

        # Evaluate the population, to ensure that the scores are obtained
        population = await self.evaluate_population(population, filtering_objectives)

        # Filter the candidates based on the validity and the three scores
        filtered_candidates = []
        for cand in population.candidates:
            if not is_valid_dna_sequence(cand.representation, self.seq_length):
                continue
            if len(filtering_objectives) > 0:
                scores = [cand.get_score(obj.name) for obj in filtering_objectives]
                if scores[max_objective_idx] != max(scores):
                    continue
            cand.representation = cand.representation.upper()
            filtered_candidates.append(cand)

        if self.use_diversity_for_filtering:
            # Filter the candidates based on the diversity
            extracted_seq_len = max(len(filtered_candidates) // 2, 1)

            dist_list = []
            for i in range(len(filtered_candidates)):
                sample_level_div = []
                for j in range(len(filtered_candidates)):
                    if i != j:
                        sample_level_div.append(hamming_distance(filtered_candidates[i].representation, filtered_candidates[j].representation))
                dist_list.append(np.mean(sample_level_div))
            top_candidated = np.argsort(dist_list)[::-1]
            filtered_candidates = [filtered_candidates[i] for i in top_candidated[0:extracted_seq_len]]
            
        return Population(candidates=filtered_candidates)

#     def _filter_candidates(self, population: Population, cell_lines_to_optimize: List[str]) -> Population:  
#         # Filter the candidates based on the validity and the three scores
#         filtered_candidates = []
#         for cand in population.candidates:
#             if not is_valid_dna_sequence(cand.representation, self.seq_length):
#                 continue
#             cand.representation = cand.representation.upper()
#             filtered_candidates.append(cand)

# #         if self.use_diversity_for_filtering:
# #             # Filter the candidates based on the diversity
# #             extracted_seq_len = max(len(filtered_candidates) // 2, 1)

# #             dist_list = []
# #             for i in range(len(filtered_candidates)):
# #                 sample_level_div = []
# #                 for j in range(len(filtered_candidates)):
# #                     if i != j:
# #                         sample_level_div.append(hamming_distance(filtered_candidates[i].representation, filtered_candidates[j].representation))
# #                 dist_list.append(np.mean(sample_level_div))
# #             top_candidated = np.argsort(dist_list)[::-1]
# #             filtered_candidates = [filtered_candidates[i] for i in top_candidated[0:extracted_seq_len]]
            
#         return Population(candidates=filtered_candidates)
    
    async def _optimize_batch(self, population: Population, objectives: List[Objective], cell_lines_to_optimize) -> Population:
        # Optimize the batch
        # Return the optimized batch
        
        if population.is_empty:
            return Population(candidates=[])
        
        population = await self.evaluate_population(population, objectives)
        prompt = self._construct_prompt(population, objectives, cell_lines_to_optimize)
        response = (await self.call_llm_with_prompt_async(prompt))['content']
        optimized_seqs = self._parse_llm_response(response)
        optimized_candidates = [Candidate(representation=seq[:self.seq_length].upper()) for seq in optimized_seqs if len(seq) >= self.seq_length]
        return Population(candidates=optimized_candidates)
        
    def _parse_llm_response(self, response: str) -> List[str]:
        # Parse the LLM response, returns the list of DNA sequences
        # The response is a string that contains the DNA sequences wrapped in <DNA> ... </DNA> tags
        # print(response)
        dna_seqs = re.findall(r'<DNA>(.*?)</DNA>', response, re.DOTALL)
        return [seq.strip() for seq in dna_seqs]

    def _construct_prompt(self, population: Population, objectives: List[Objective], cell_lines_to_optimize: List[str]) -> str:
        """Construct the prompt for the LLM"""

        if len(cell_lines_to_optimize) == 0:
            raise ValueError("No cell lines to optimize for")
        if len(cell_lines_to_optimize) > 1:
            raise NotImplementedError("More than one cell line to optimize for, which is not supported yet")
        cell_line = cell_lines_to_optimize[0]

        if cell_line.lower() == 'hepg2':
            cell_line = 'HepG2'
        elif cell_line.lower() == 'k562':
            cell_line = 'K562'
        elif cell_line.lower() == 'sknsh' or cell_line == 'SK-N-SH':
            cell_line = 'SKNSH'
        else:
            raise ValueError(f"Invalid cell line: {cell_line}")

        # Use ObjectiveIndex to quickly organize and retrieve objectives
        objective_index = ObjectiveIndex(objectives)

        # Some easy-to-access variables
        num_seqs = population.size
        seq_length = self.seq_length
        
        # === Construct specific_task_instruction ===
        maximization_objectives = objective_index.get_maximization_objectives()
        minimization_objectives = objective_index.get_minimization_objectives()

        maximization_objective_names = [obj.name for obj in maximization_objectives]
        minimization_objective_names = [obj.name for obj in minimization_objectives]

        specific_task_instruction = f"Specifically, the proposed DNA sequences should have higher scores for {', '.join(maximization_objective_names)}, and lower scores for {', '.join(minimization_objective_names)}.".strip()

        # === Construct regular_objective_names_str ===
        regular_objectives = objective_index.get_all_regular()
        regular_objective_names = [obj.name for obj in regular_objectives]
        regular_objective_names_str = ", ".join(regular_objective_names)

        # === Construct inputs ===
        input_template = "DNA: {dna_seq}    All objectives: {properties_scores_str}"
        inputs = []
        all_scores = []
        for cand in population.candidates:
            scores = []
            for obj_name in regular_objective_names:
                scores.append(cand.get_score(obj_name))
            all_scores.append(scores)
            properties_scores_str = ", ".join('%.3f' % score for score in scores)
            inputs.append(
                input_template.format(dna_seq=cand.representation, properties_scores_str=properties_scores_str)
            )

        # avg scorers
        all_scores = np.array(all_scores)
        avg_scores = np.mean(all_scores, axis=0)
        avg_scores_str = ", ".join('%.3f' % score for score in avg_scores)
        avg_scores_str = f"Average scores for the batch: {avg_scores_str}"

        # inputs str
        inputs_str = "\n".join(inputs) + "\n\n" + avg_scores_str

        # === Construct population_wise_objective_str ===
        population_wise_objectives = objective_index.get_all_population_wise()
        if len(population_wise_objectives) > 0:
            population_wise_objective_scores = {}
            for obj in population_wise_objectives:
                obj_name = obj.name
                obj_score = population.get_score(obj_name)
                population_wise_objective_scores[obj_name] = obj_score
            population_wise_objective_str = "\n".join([f"{obj_name}: {obj_score}" for obj_name, obj_score in population_wise_objective_scores.items()])
            population_wise_objective_str = f"The properties of the whole batch of DNA sequences include:\n{population_wise_objective_str}"
        else:
            population_wise_objective_str = ""
            
        prompt = DNA_OPTIMIZATION_PROMPT_NODIV_TEMPLATE.format(
            num_seqs=num_seqs,
            seq_length=seq_length,
            cell_line=cell_line,
            specific_task_instruction=specific_task_instruction,
            regular_objective_names_str=regular_objective_names_str,
            inputs_str=inputs_str,
            population_wise_objective_str=population_wise_objective_str
        )

        # To Tianyu: This implementation (adding stability enhanced prompts) is ugly. Can we just use one prompt for all cases, requiring diversity and stability regardless of what is in your self.REQUIRED_OBJECTIVE_NAMES? Would it deteriorate the performance?
        for key in self.REQUIRED_OBJECTIVE_NAMES:
            # use stability enhanced prompts for design
            if 'stability' in key.lower():
                prompt = DNA_OPTIMIZATION_PROMPT_TEMPLATE.format(
                    num_seqs=num_seqs,
                    seq_length=seq_length,
                    cell_line=cell_line,
                    specific_task_instruction=specific_task_instruction,
                    regular_objective_names_str=regular_objective_names_str,
                    inputs_str=inputs_str,
                    population_wise_objective_str=population_wise_objective_str
                )
                break
        
        # print(prompt)   # To Tianyu: LLM calling will be printed out with the DEBUG logging level, so we don't need to print it here. Use logger to print anything you want instead of `print()`
        
        return prompt
