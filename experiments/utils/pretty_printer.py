import json


def cleanup_triples(triples: list[dict]):
    res = []
    for triple in triples:
        v_triple = " ".join([triple['sub'], triple['rel'], triple['obj']]).lower()
        if not (v_triple in res):
            res.append(v_triple)
    return res

if __name__ == '__main__':
    data = []
    with open('../results/llm_responses/Babelscape.rebel-large-12-beams/ont_1_movie-wikidata_tekgen.jsonl') as f:
        data = [json.loads(line) for line in f]


    for response in data:
        response['triples'] = cleanup_triples(response['triples'])

        print({
            'id': response['id'],
            'triples': response['triples']
        })