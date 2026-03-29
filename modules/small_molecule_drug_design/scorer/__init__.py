# """Scorer package for small molecule drug design."""

# Ensure scorers register themselves via import side effects
# from . import unidock_scorer  # noqa: F401
# NOTE: Comment this line out temporarily if you do not have antibiotic model files that ChempropScorers depends on
# from . import chemprop_scorer  # noqa: F401  # Temporarily disabled due to torch import issues
from . import small_world_similarity_scorer  # noqa: F401

# from . import ra_scorer  # noqa: F401
