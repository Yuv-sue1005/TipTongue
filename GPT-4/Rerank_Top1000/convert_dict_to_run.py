import pickle
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    parser.add_argument('--input', default='top1000.test.reranked.pkl', type=str, help='input document to convert into a run')
    parser.add_argument('--output', required=True, type=str, help='output file')
    args = parser.parse_args()
    return args


args = parse_args()

titles = pickle.load(open("../../WIKIPEDIA/wikipedia_titles.pkl".format(args.domain), 'rb'))
titles_to_dids = {v: k for k, v in titles.items()}

#test-culture-ahrtsdlgra-con01a Q0 test-culture-ahrtsdlgra-con01a 1 319.341888 Anserini

queries = pickle.load(open("../../DATA/{}_queries.pkl".format(args.domain), 'rb'))
qrels = pickle.load(open("../../DATA/test_{}_qrels_human.pkl".format(args.domain), 'rb'))
queries = {qid: queries[qid] for qid in qrels}
results = pickle.load(open(args.input, 'rb'))

with open(args.output, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter=' ')

    for qid in queries:
        ranks = results[qid]
        for i, title in enumerate(ranks):
            tsv_writer.writerow([qid, 'Q0', titles_to_dids[title], str(i + 1), float(1000 - i), 'Anserini'])

