#!/bin/sh
python3 training.py doc2vec-bow
python3 training.py bow
python3 training.py doc2vec

