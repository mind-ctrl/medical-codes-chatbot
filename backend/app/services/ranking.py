"""
Ranking and Fusion Services
Implements Reciprocal Rank Fusion for combining search results
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60,
    key_field: str = 'cpt_code'
) -> List[Dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion

    RRF Formula: score(d) = Σ(1 / (k + rank(d)))

    Why RRF works:
    - Handles different score scales (similarity vs rank)
    - Gives bonus to items appearing in multiple lists
    - Constant k=60 is empirically optimal

    Example:
    Vector search:  E11.9 (rank 1), E11.65 (rank 2)
    Keyword search: E11.9 (rank 2), E11.8 (rank 1)

    RRF scores:
    - E11.9: 1/61 + 1/62 = 0.0327  (appears in both, ranked high)
    - E11.65: 1/62 = 0.0161
    - E11.8: 1/61 = 0.0164

    Final ranking: E11.9, E11.8, E11.65

    Args:
        results_lists: List of result lists (each from different search method)
        k: RRF constant (default 60, from research)
        key_field: Field to use as unique identifier

    Returns:
        Combined and re-ranked results with rrf_score
    """
    # Collect all unique items with their RRF scores
    rrf_scores: Dict[str, Dict] = {}

    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            key = item[key_field]

            if key not in rrf_scores:
                # First time seeing this item
                rrf_scores[key] = {
                    **item,
                    'rrf_score': 0,
                    'component_scores': []
                }

            # Add RRF score contribution from this ranking
            score = 1.0 / (k + rank)
            rrf_scores[key]['rrf_score'] += score
            rrf_scores[key]['component_scores'].append(score)

    # Sort by RRF score (highest first)
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    return ranked_results


def normalize_scores(
    results: List[Dict],
    score_field: str = 'rrf_score'
) -> List[Dict]:
    """
    Normalize scores to 0-1 range for confidence scores

    Min-max normalization:
    - Highest score → 1.0
    - Lowest score → 0.0
    - Everything else scaled proportionally

    Why normalize:
    - RRF scores are arbitrary (depend on k and number of lists)
    - Users understand 0.95 confidence better than 0.0327 RRF score
    - Makes scores comparable across different queries

    Args:
        results: List of results with scores
        score_field: Field containing the score to normalize

    Returns:
        Results with added confidence_score field (0-1 range)
    """
    if not results:
        return results

    max_score = max(item[score_field] for item in results)
    min_score = min(item[score_field] for item in results)

    # Avoid division by zero if all scores are identical
    range_score = max_score - min_score if max_score != min_score else 1.0

    for item in results:
        item['confidence_score'] = (item[score_field] - min_score) / range_score

    return results
