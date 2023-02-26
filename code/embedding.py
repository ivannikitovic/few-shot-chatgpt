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

if __name__ == "__main__":
    if not os.path.exists(embeddings_location):
        make_embeddings(embedding_engine, embeddings_location)