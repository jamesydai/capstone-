#!/usr/bin/env python

import sys
import json

from src.data.etl import get_data
from src.features.build_features import get_features
from src.models.modeling import perform_modeling
from src.visualizations.visualize import create_visualizations

def main(targets):
    if "test" in targets:
        with open("test-params.json") as fh:
            data_params = json.load(fh)
    else:
        with open("data-params.json") as fh:
            data_params = json.load(fh)
    get_data(**data_params)
    get_features(**data_params)
    perform_modeling(**data_params)
    create_visualizations(**data_params)

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
