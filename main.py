import pandas as pd
import numpy as np

from on_demand_parsing.parser import run_parser
from preprocessing.create_features import build_real_features

query = input("Введите запрос: ")
run_parser(query)

build_real_features("data/openalex_corpus.parquet", "data/processed_ml_dataset.parquet")
