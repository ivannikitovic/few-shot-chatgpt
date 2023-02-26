import openai
# import umap
import json
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import util
import os

openai.api_key = os.getenv('OpenAI_API_Key')

embeddings_location = 'data/question_embeddings.json'
embedding_engine = 'text-embedding-ada-002'

num_questions = 22

def make_embeddings(embedding_engine, embeddings_location):
    """
    Takes json files of questions using our json file formatting, 
        embeds them using OpenAI's embedding_engine,
        and saves a new json, embeddings.json, of the embeddings.
    """
    list_of_embeddings = []

    for num in range(1, num_questions + 1):
        json_location = 'data/Question_' + str(num) + '.json'
        with open(json_location, 'r') as f:
            data = json.load(f)
        raw_question = data['Question']
        embedding = openai.Embedding.create(input = raw_question, 
                                            engine = embedding_engine)['data'][0]['embedding']
        list_of_embeddings.append(embedding)

    embeddings = {'list_of_embeddings':list_of_embeddings}
    with open(embeddings_location, 'w') as f:
        f.write(json.dumps(embeddings))

def add_similar_sorted(embeddings):
    """
    Takes a list of embeddings and appends the most similar questions
        using cosine similarity to the Question json files.
    """

    for num in range(1, num_questions + 1):
        json_location = 'data/Question_' + str(num) + '.json'

        sorted_similar_questions = get_most_similar(embeddings, num-1)

        with open(json_location, 'r') as f:
            data = json.load(f)
            data['Similar_Questions'] = sorted_similar_questions

        os.remove(json_location)
        with open(json_location, 'w') as f:
            json.dump(data, f, indent=4)

def get_most_similar(embeddings, i):
    """
    Returns most similar questions, while they are in their embedded form, 
        to the target, index i, via cosine similarity.
    """
    cos_sims = []
    cos_to_num = {}
    for j in range(len(embeddings)):
        cos_sim = util.cos_sim(embeddings[i], embeddings[j]).item()
        cos_to_num[cos_sim] = j
        cos_sims.append(cos_sim)
    ordered = sorted(cos_sims, reverse=True)
    closest_qs = []
    for val in ordered:
        closest_qs.append(cos_to_num[val]+1)
    return closest_qs[1:]

def get_embeddings(embeddings_location):
    """
    Retrieves embeddings from embeddings_file. 
        Embeddings are assumed to be (n x d).
    """
    with open(embeddings_location, 'r') as f:
        points = json.load(f)['list_of_embeddings']
    return np.array(points)

if __name__ == "__main__":
    if not os.path.exists(embeddings_location):
        make_embeddings(embedding_engine, embeddings_location)

    embeddings = get_embeddings(embeddings_location)
    add_similar_sorted(embeddings)