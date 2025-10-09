"""
Enhanced refinement wrapper for the DSV framework.
Provides a small, well-structured class that reuses DSVFramework and adds
an iterative refinement context with example retrieval support.
"""

import time
import logging
"""
Enhanced refinement wrapper for the DSV framework.
This file provides a small, well-structured class that reuses DSVFramework and
offers a conservative refinement loop entry point without changing core logic.
"""

from typing import List, Dict, Any, Optional
"""
Enhanced refinement wrapper for the DSV framework.
This file provides a small, well-structured class that reuses DSVFramework and
offers a conservative refinement loop entry point without changing core logic.
"""

from dataclasses import dataclass

from dsv_framework import DSVFramework, DSVProcessResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class RefinementContext:
    iteration: int
    previous_results: List[Dict[str, Any]]
    failure_reasons: List[str]
    similarity_scores: List[float]
    best_result: Optional[Dict[str, Any]]
    refinement_strategy: str


class EnhancedRefinementDSV:
    """Lightweight enhanced refinement wrapper around DSVFramework."""

    def __init__(self, config_path: str = "config/dsv_config.json"):
        self.dsv = DSVFramework(config_path=config_path)
        # forward a couple of convenient settings
        self.max_refinement_iterations = self.dsv.max_refinement_iterations
        self.similarity_threshold = self.dsv.similarity_threshold

    def _determine_refinement_strategy(self, context: RefinementContext) -> str:
        if not context.previous_results:
            return "standard"
        avg = sum(context.similarity_scores) / len(context.similarity_scores) if context.similarity_scores else 0.0
        if avg < 0.3:
            return "re-extract atomic propositions and focus on core events"
        if avg < 0.6:
            return "refine temporal relations and metric constraints"
        return "tune MTL operator choices and time windows"

    def process_with_enhanced_refinement(self, sentence: str, enable_refinement: bool = True) -> DSVProcessResult:
        """Run the underlying DSV process with a simple, conservative refinement wrapper.

        Currently this wrapper calls the core DSVFramework.process() and, on failure,
        optionally retries once. Future work can inject additional context or alter
        prompts between attempts.
        """
        start_time = time.time()
        logger.info("Starting enhanced refinement DSV processing")

        context = RefinementContext(
            iteration=0,
            previous_results=[],
            failure_reasons=[],
            similarity_scores=[],
            best_result=None,
            refinement_strategy="standard"
        )

        # Primary run
        result = self.dsv.process(sentence, enable_refinement=enable_refinement)

        # If it failed and refinement is enabled, perform a single conservative retry.
        if not result.success and enable_refinement and result.refinement_iterations < self.max_refinement_iterations:
            context.iteration = 1
            context.refinement_strategy = self._determine_refinement_strategy(context)
            logger.info("Retrying with refinement strategy: %s", context.refinement_strategy)
            result = self.dsv.process(sentence, enable_refinement=enable_refinement)

        result.total_processing_time = time.time() - start_time
        return result


def main():
    print("EnhancedRefinementDSV demo")
    er = EnhancedRefinementDSV()
    test = "Within 5 to 10 seconds after sensor A detects a fault, alarm B must sound for at least 20 seconds."
    res = er.process_with_enhanced_refinement(test)
    print("Success:", res.success)
    print("Final formula:", res.final_mtl_formula)


if __name__ == '__main__':
    main()
