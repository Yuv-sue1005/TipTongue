from openai import AzureOpenAI

import pickle
import argparse
from difflib import get_close_matches

client = AzureOpenAI(
    api_key="",  # Your api key here
    api_version="2024-05-01-preview",
    azure_endpoint = "" # Your azure end_point
    )




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all', type=str,
                        help='Domain to train on')
    parser.add_argument('--model_name', default="gpt-4-1106-preview", type=str, help="name of model on azure")
    parser.add_argument('--input_run', required=True, help="round robin pkl file outputed by round_robin.py")
    args = parser.parse_args()
    return args


args = parse_args()

model_name = args.model_name

assistant = client.beta.assistants.create(
    name="Re-Ranker",
    instructions="""
    You are a helpful assistant, and you would help improve an exsisting ranking based on the description you are given.
    """,
    model=model_name  # Replace this with the actual deployment name of your model
)


def prompt_model(prompt, model_name, assistant=assistant):
    response = assistant.chat.create(
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

queries = pickle.load(open("queries.pkl", 'rb'))
titles = pickle.load(open("query_titles.pkl", 'rb'))
query_titles = pickle.load(open("../../DATA/{}_titles.pkl".format(args.domain), 'rb'))
qrel = pickle.load(open(".qrels.pkl", 'rb'))
ranks = pickle.load(open(args.input_run, 'rb'))
queries = {qid:queries[qid] for qid in qrel}

answers = {}

K = 100

true = pickle.load(open("top{}.test.reranked.pkl".format(K), 'rb'))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def convert_to_list(answer):
    items = []

    split = answer.split("\n")

    for i, string in enumerate(split):
        if string.startswith("1"):
            true_items = split[i:]
            break
    try:
        for item in true_items:
            it = item.split(".")
            it = '.'.join(it[1:])
            if it[0] == ' ':
                it = it[1:]
            if it[-1] == ' ':
                it = it[:-1]
            items.append(it)
    except:
        return []

    return items


def rerank(query, item_list):
    prompt = "I am going to give you a question and a list of items. Re-order the items according to the likelihood that the question refers to the item.  " \
             "Format the answer as a numbered list of {} items. Keep the item names I give you. " \
             "Be direct and return the list ONLY. Stick with items from the list ONLY.\n\nQUESTION: {}\n\nITEM LIST: {}".format(
        K, query, '\n'.join(item_list))

    succeeded = False
    attempts = 0
    while not succeeded and attempts < 3:
        try:
            answer = prompt_model(prompt, model_name)
            succeeded = True
        except:
            succeeded = False
            attempts = attempts + 1

    if not succeeded:
        print("FAILED")
        return []

    llm_list = convert_to_list(answer)

    if llm_list == []:
        print(answer)
        return []

    candidates = item_list.copy()

    # Build re-ranked list #
    reranked_list = []
    for title in llm_list:

        if title in reranked_list:
            continue

        if title in candidates:
            reranked_list.append(title)
            candidates.remove(title)
        else:
            closest = get_close_matches(title, candidates, cutoff=0.0)[0]
            print("Title: {}".format(title))
            print("Closest: {}".format(closest))
            print("Candidates: {}".format(candidates))
            reranked_list.append(closest)
            candidates.remove(closest)

    # If re-ranked list does not contain K elements, need to add remaining by the same order #
    if len(candidates) > 0:
        reranked_list = reranked_list + candidates

    return reranked_list


for i, qid in enumerate(queries):
    if qid not in true:
        print(i, flush=True)

        query = query_titles[qid] + ' ' + queries[qid]

        top1000 = ranks[qid][:K]

        #items = [titles[pid] for pid in top1000]
        items = [title for title in top1000]

        if titles[qrel[qid]] not in top1000:
            answers[qid] = items
            continue

        answers[qid] = rerank(query, items)

    else:
        answers[qid] = true[qid]

pickle.dump(answers, open("top{}.test.reranked.pkl".format(K), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
