import pickle
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str, help='Domain to train on')
    parser.add_argument('--retrieve_results', type=str, help='top 1k docs to rerank')
    parser.add_argument('--outputs', type=str, help='output doc')
    args = parser.parse_args()
    return args

args = parse_args()

# {"qid": ["rank", "docid"]}

ranks = pickle.load(open(os.path(args.retrieve_results), 'rb'))

new_ranks = {qid:[[] for _ in range(10)] for qid in ranks}

for qid in ranks:

    bucket = 0
    for rank,docid in enumerate(ranks[qid]):

        new_ranks[qid][bucket].append(docid)
        bucket = bucket +  1
        if bucket == 10:
            bucket = 0

merged_ranks = {qid:[] for qid in ranks}

for qid in new_ranks:

    for bucket in new_ranks[qid]:

        merged_ranks[qid] = merged_ranks[qid] + bucket


pickle.dump(merged_ranks, open(os.path(args.output), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


