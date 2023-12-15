import pandas as pd
import numpy as np
import sys, json, re, time
import matplotlib.pyplot as plt
from numpy.linalg import norm
from statistics import mean
from itertools import combinations
from sklearn.utils import resample
from sklearn.cluster import AgglomerativeClustering

with open('/Users/kasperdekoning/Downloads/TVs-all-merged.json') as file:
    tv_data = json.load(file)

# Separate tv model ids from the data
proxy = [val for sublist in tv_data.values() for val in sublist]
tv_df = pd.DataFrame(proxy)  # df

def cosine_distance(x, y):
    similarity = np.dot(x, y) / (norm(x) * norm(y))
    distance = 1 - similarity
    return distance

def minhashing(input, n, p):
    permutation_vector_a = np.random.permutation(n)
    permutation_vector_b = np.random.permutation(n)
    rows, cols = input.shape
    signature_matrix = np.full((n, cols), sys.maxsize)
    for r in range(len(input)):
        for c in range(len(input[0])):
            if input[r][c]:
                # Calculate the hash values next with the hash function: (a + (b * x)) % p
                hashvalues = [(permutation_vector_a[k] + permutation_vector_b[k] * r) % p for k in range(n)]
                signature_matrix[:, c] = np.minimum(signature_matrix[:, c], hashvalues)
    return signature_matrix



resolutions = ['720p', '1080p', '4K']
refresh_rates = ['50/60hz', '60hz', '120hz', '240hz', '600hz']
unique_tv_brands = ['affinity', 'avue', 'azend', 'coby', 'compaq', 'contex', 'craig', 'curtisyoung', 'dynex', 'elo',
                    'epson', 'gpx', 'haier', 'hannspree', 'hiteker', 'hisense', 'insignia', 'jvc', 'lg', 'magnavox',
                    'mitsubishi', 'naxa', 'nec', 'optoma', 'panasonic', 'philips', 'proscan', 'pyle', 'rca',
                    'samsung', 'sanyo', 'sansui', 'seiki', 'sharp', 'sceptre', 'sigmac', 'sony', 'sunbritetv',
                    'supersonic', 'tcl', 'toshiba', 'upstar', 'venturer', 'venturer', 'viewsonic', 'vizio',
                    'westinghouse']


def clean_data(data):
    """
    This function cleans the data acccordingly and standardizes notations of certain key attributes
    """
    model_id, shop, title = data['modelID'], data['shop'], data['title']

    # Titles
    title = title.str.lower()
    title = title.str.replace(r'[.:;+\[\]()]|', '', regex=True)
    title = title.replace('"', 'inch').replace(' inch', 'inch').replace('inch ', 'inch').replace(
        '-inch', 'inch').replace('Inch', 'inch').replace('-Inch', 'inch').replace(' Inch', 'inch').replace(' inches',
                                                                                                           'inch').replace(
        'inches ', 'inch').replace('in', 'inch')
    title = title.replace('HZ', 'HZ').replace('Hz', 'hz').replace('hertz', 'hz').replace(
        ' hertz', 'hz').replace('Hertz', 'hz').replace('-Hertz', 'hz').replace('-hertz', 'hz').replace(
        ' hz', 'hz').replace('hZ', 'hz').replace(' Hz', 'hz')

    model_ids = [re.sub(r'[^\w\s]', '', str(mid)).lower() for mid in model_id]

    titles = [re.sub(r'[^\w\s]', '', str(t)).lower().split() for t in title]

    key_titles = []
    seen = set()
    for sublist in titles:
        for title in sublist:
            if title not in seen:
                key_titles.append(title)
                seen.add(title)

    tv_resolution = [next((j for j, resolution in enumerate(resolutions) if resolution in title), 0) for title in
                     titles]
    tv_refresh_rate = [next((j for j, rate in enumerate(refresh_rates) if rate in title), 0) for title in titles]
    tv_brand = [next((j for j, brand in enumerate(unique_tv_brands) if brand in title), 0) for title in titles]

    cleaned_data = pd.DataFrame({
        'modelID': model_ids,
        'title': titles,
        'shop': shop,
        'brand': tv_brand,
        'resolution': tv_resolution,
        'refresh rate': tv_refresh_rate
    })

    return cleaned_data, key_titles

def create_binary_matrix(titles, key_titles):
    """
    Creates a characteristic matrix. Has binary values that represent the presence (1) or absence (0)
    of certain key value terms in the tile
    """
    return np.array([[1 if key in title else 0 for title in titles] for key in key_titles])


def signature_matrix(data, keytitles, bands, rows):
    """
    Creates a signature matrix
    """
    p = 9787
    n = bands * rows
    titles = data['title']
    input_matrix = create_binary_matrix(titles, keytitles)
    signature_matrix = minhashing(input_matrix, n, p)

    # Banding
    hash_bands = []
    for i in range(bands):
        band = signature_matrix[i * rows:(i + 1) * rows]
        band_hashes = [''.join(map(str, map(int, band[:, j]))) for j in range(band.shape[1])]
        hash_bands.append(band_hashes)
    return input_matrix, signature_matrix, hash_bands

def candidate_matrix(band_list, bands):
    """
    Creates a matrix with candidate pairs
    """
    n = len(band_list[0])
    candidate = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            if any(band_list[c][i] == band_list[c][j] for c in range(bands)):
                candidate[i, j] = candidate[j, i] = 1

    return candidate

def dissimilarity_matrix(data, candidate, input_matrix):
    """
    Create a dissimilarity matrix based on the cosine distance
    """
    brand, resolution, shop, refresh_rate = data['brand'], data['resolution'], data['shop'], data['refresh rate']
    n = len(candidate)
    dis_matrix = np.ones((n, n)) * sys.maxsize


    for i in range(n):
        for j in range(i + 1, n):
            if (brand[i] == brand[j] and
                    resolution[i] == resolution[j] and
                    shop[i] != shop[j] and
                    refresh_rate[i] == refresh_rate[j] and
                    candidate[i, j] == 1):
                distance = cosine_distance(input_matrix[:, i], input_matrix[:, j])
                dis_matrix[i, j] = dis_matrix[j, i] = distance

    dis_matrix[dis_matrix == 0] = np.inf
    return dis_matrix


def get_actual_duplicates(data):
    """
    Retrieve the pairs that are actual duplicates
    """
    tv_ids = data['modelID']
    actual_duplicates = set()
    for modelID in set(tv_ids):
        duplicate = np.where(modelID == tv_ids)[0]
        if len(duplicate) > 1:
            actual_duplicates.update(combinations(duplicate, 2))
    actual_duplicates = list(actual_duplicates)

    return actual_duplicates


def get_potential_duplicates(dis_matrix, threshold):
    """
    Retrieve predicted duplicate pairs using hierarchical clustering.
    """
    # Initialize hierarchical clustering

    hierarchical_clustering = AgglomerativeClustering(metric='precomputed',
                                                      linkage='complete',
                                                      distance_threshold=threshold,
                                                      n_clusters=None)
    hierarchical_clustering.fit(dis_matrix)

    # Finding groups of similar items
    predicted_duplicates = [group for cluster_id in range(hierarchical_clustering.n_clusters_)
                            for group in combinations(np.where(hierarchical_clustering.labels_ == cluster_id)[0], 2)
                            if len(group) > 1]

    return predicted_duplicates


def calculate_results(data, bands, rows, threshold):
    """
    Calculates the performance measures
    """
    cleaned_data, key_titles = clean_data(data)
    binary_matrix, _, band_matrix = signature_matrix(cleaned_data, key_titles, bands, rows)

    candidates = candidate_matrix(band_matrix, bands)
    dissim_matrix = dissimilarity_matrix(cleaned_data, candidates, binary_matrix)

    predicted_duplicates = get_potential_duplicates(dissim_matrix, threshold)
    actual_duplicates = get_actual_duplicates(data)

    true_positives = [dupe for dupe in predicted_duplicates if dupe in actual_duplicates]
    false_positives = [dupe for dupe in predicted_duplicates if dupe not in actual_duplicates]

    n_true_positives = len(true_positives)
    n_false_positives = len(false_positives)
    n_false_negatives = len(actual_duplicates) - n_true_positives

    precision = n_true_positives / (n_true_positives + n_false_positives)
    recall = n_true_positives / (n_true_positives + n_false_negatives)

    n_comparisons = np.count_nonzero(candidates) / 2
    total_possible_comparisons = len(data) * (len(data) - 1) / 2
    fractions_of_comparisons = n_comparisons / total_possible_comparisons

    pq = n_true_positives / n_comparisons
    pc = n_true_positives / len(actual_duplicates)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_star = 2 * pq * pc / (pq + pc)


    #print(f"Number of predicted duplicates: {len(predicted_duplicates)}")
    #print(f"True positives: {n_true_positives}")
    #print(f"Pair quality: {pq}")
    #print(f"Pair completeness: {pc}")
    #print(f"F1 Score: {f1_score}")
    #print(f"F1 Star: {f1_star}")
    #print(f"Fraction of comparisons: {comparison_fraction}")

    return len(predicted_duplicates), n_true_positives, f1_score, f1_star, pc, pq, fractions_of_comparisons


def bootstrapping(data, num_bootstraps, bands, rows, threshold):
    """
    Conduct bootstrapping
    """
    metrics = np.zeros((num_bootstraps, 7))

    for i in range(num_bootstraps):
        bootstrap_sample = resample(data[['modelID', 'title', 'shop']], n_samples=len(data), random_state=i)
        train_set = bootstrap_sample.drop_duplicates()

        # Construct the test set from rows in df not present in the train_set and reset index
        test_set = data.loc[~data.index.isin(train_set.index)].reset_index(drop=True)

        # Evaluate duplicates
        metrics[i, :] = calculate_results(test_set, bands, rows, threshold)

    # Calculate and print average values for each metric
    metric_names = ["Number of predicted duplicates", "True positives", "f1 score",
                    "f1 star", "Pair completeness", "Pair quality", "fractions of comparisons"]
    for j, name in enumerate(metric_names):
        print(f"{name} average: {mean(metrics[:, j])}")

    return tuple(metrics[:, j] for j in range(7))


# Initialize lists to store the results
bootstrap_results = []
fractions_of_comparisons, f1_score, f1_star, pq, pc = ([] for i in range(5))

# Define the different values for the parameters: (bands, rows)
bootstrap_parameters = [(700, 1), (350, 2), (233, 3), (175, 4), (140, 5)]

# Loop through each set of parameters and perform bootstrapping
for bands, rows in bootstrap_parameters:
    result = bootstrapping(tv_df, 5, bands, rows, 0.50)
    bootstrap_results.append(result)

# Calculate the mean for each metric from the results
for result in bootstrap_results:
    f1_score.append(mean(result[2]))
    f1_star.append(mean(result[3]))
    pc.append(mean(result[4]))
    pq.append(mean(result[5]))
    fractions_of_comparisons.append(mean(result[6]))


# Creating plots based on the results

plt.plot(fractions_of_comparisons, f1_score, linewidth=2, color='navy', marker='o')
plt.grid()
plt.title('F1 score', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('f1_score')
plt.show()

plt.plot(fractions_of_comparisons, f1_star, linewidth=2, color='navy', marker='o')
plt.grid()
plt.title('F1 star', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('f1_star')
plt.show()


plt.plot(fractions_of_comparisons, pc, linewidth=2, color='navy', marker='o')
plt.grid()
plt.title('Pair completeness', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('pc')
plt.show()

plt.plot(fractions_of_comparisons, pq, linewidth=2, color='navy', marker='o')
plt.grid()
plt.title('Pair quality', fontsize=16, color='navy')
plt.xlabel('Fractions of comparisons')
plt.ylabel('pq')
plt.show()

