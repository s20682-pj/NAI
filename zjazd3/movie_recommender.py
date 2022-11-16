"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python project to show recommended and not recommended movies to watch based on what have been watched
and what other users watched and scored
System requirements:
- Python 3.10
- json
- argparse
- Numpy
"""
import argparse
import json
import numpy as np

"""
Adding argument for which user recommendations will be indicated
"""
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

"""
Computing the Pearson correlation score between user1 and user2;
Checking what movies both user watched, and how's scored them, 
then calculating similarity between users to determine which movies can be enjoyed and not for given user
"""
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

"""
Choosing movies from dataset based on Pearson score
"""
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue
        
        filtered_list = [x for x in dataset[user] if x not in \
                dataset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list: 
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Sort in decreasing order
    overall_scores = sorted(overall_scores.items(), key=lambda item: item[1], reverse=True)

    # Extract the movie recommendations
    movie_recommendations = [movie for movie, _ in overall_scores]

    return movie_recommendations

"""
Get all unique movies from dataset without movies that chosen user saw
"""
def get_movies(dataset, input_user):
    unique_movies = set()
    for key in dataset:
        for movie in dataset[key]:
            if movie not in dataset[input_user].keys():
                unique_movies.add(movie)
    return unique_movies
 
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nMovie recommendations for " + user + ":")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies[0:5]):
        print(str(i+1) + '. ' + movie)

    all_movies = get_movies(data, user)
    not_recommended_movies = all_movies.difference(movies)

    print("\nMovies not recommended for " + user + ":")
    for i, nrm in enumerate(list(not_recommended_movies)[0:5]):
        print(str(i+1) + '. ' + nrm)
