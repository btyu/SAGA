#!/bin/bash
SCORER_FOLDER=$(basename "$(dirname "$(realpath "$0")")")
echo "Importing the scorer..."
echo "SCORER_FOLDER: ${SCORER_FOLDER}"
python -c "from ${SCORER_FOLDER}.base import Scorer; scorer = Scorer(); print('Test passed!')"


