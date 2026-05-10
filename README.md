# Music Recommendation System

A simple music recommender system that suggests music artists to users based on their listening history.

## Project Idea

This project uses collaborative filtering to recommend artists.  
The first version uses Alternating Least Squares, also known as ALS, with the `implicit` Python library.

The goal is to learn how music recommendation systems work and build a clean project that can be improved later.

## Features

- Recommend artists for a user
- Find similar artists
- Use implicit feedback such as play counts
- Train an ALS collaborative filtering model
- Evaluate recommendation quality

## Tech Stack

- Python
- Pandas
- NumPy
- Scipy
- implicit
- Jupyter Notebook

## How It Works

1. Load user-artist listening data
2. Convert the data into a sparse user-item matrix
3. Train an ALS model
4. Generate artist recommendations for each user
5. Evaluate the results using recommendation metrics

## Project Structure

```text
music-recommender/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── recommend.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md