# Utility script to generate sentences from triples by building
# a prompt for an OpenAI GPT. A random, connex subset of the 
# available triples is chosen to provide in the prompt for the model. 
import os
import sys
import json
import random
from openai import OpenAI
from multiprocessing.pool import ThreadPool


class SentenceGenerator:
    """
    >>> from gen_sent import SentenceGenerator
    >>> lotr = {
            'triples': [
                {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "director", "rpid": "P57", "obj": "Peter Jackson", "oqid": "Q4465"}, 
                {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"},
                {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Orlando Bloom", "oqid": "Q44467"},
                {"sub": "Orlando Bloom", "sqid": "Q44467", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"},
                {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Gollum", "oqid": "Q15007"},
                {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Frodo Baggins", "oqid": "Q177329"},
            ]
        }
    >>> metropolis = {
            'triples': [
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "director", "rpid": "P57", "obj": "Fritz Lang", "oqid": "Q19504"}, 
                {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "Austria", "oqid": "Q40"},
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "publication date", "rpid": "P577", "obj": "1927-01-10T00:00:00Z", "oqid": "1927-01-10T00:00:00Z"},  
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "filming location", "rpid": "P915", "obj": "Berlin", "oqid": "Q64"}, 
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "love", "oqid": "Q316"}, 
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "revolution", "oqid": "Q10931"}, 
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "technology", "oqid": "Q11016"}, 
                {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "working class", "oqid": "Q191159"}
            ]
        }
    >>> gen = SentenceGenerator([lotr, metropolis], '', 'movies', 'ont_1_movie')
    >>> gen.generate(stdout=True)
    T0 Getting prompt and triples for ont_1_movie_train_Q164963
    n_triples 9
    {
        "id": "ont_1_movie_train_Q164963", 
        "sent": '''Peter Jackson, a director from New Zealand, masterfully helmed The Lord of the Rings: The Two Towers. The film features 
        an impressive cast, including Orlando Bloom, who hails from the United Kingdom. In this epic sequel, audiences are introduced 
        to pivotal characters such as Frodo Baggins and the complex Gollum. Their intertwined destinies drive the narrative forward, showcasing 
        the film's rich storytelling.''', 
        "triples": [
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "director", "rpid": "P57", "obj": "Peter Jackson", "oqid": "Q4465"}, 
            {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            ...
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Frodo Baggins", "oqid": "Q177329"}
        ]
    }
    T0 Getting prompt and triples for ont_1_movie_train_Q151599
    n_triples 6
    {
        "id": "ont_1_movie_train_Q151599", 
        "sent": '''Fritz Lang, an Austrian director, helmed the groundbreaking film Metropolis, which explores themes of love and the struggles of the working class. 
        Set against the backdrop of Berlin, this cinematic masterpiece depicts a dystopian society on the brink of revolution, intertwining personal relationships 
        with social commentary. Through its depiction of love amid societal upheaval, Metropolis remains a significant work in the genre of science fiction.''', 
        "triples": [
            {"sub": "Metropolis", "sqid": "Q151599", "rel": "director", "rpid": "P57", "obj": "Fritz Lang", "oqid": "Q19504"}, 
            {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "love", "oqid": "Q316"}, 
            ...
            {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "revolution", "oqid": "Q10931"}
        ]
    }
    """
    def __init__(self, samples: list[dict[str, str|list[dict]]], output_file_path: str, prompt_subject: str, ont_name: str):
        self.prompt_subject =  prompt_subject
        self.output_file_path = output_file_path
        self.samples = samples
        self.ont_name = ont_name
    
    def generate(self, i: int = 0, stdout: bool = False):
        client = OpenAI(api_key="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")

        for sample in self.samples:
            sample['id'] = f"{self.ont_name}_train_{sample['triples'][0]['sqid']}"
            
            print(f"T{i} Getting prompt and triples for", sample['id'])
            prompt, chosen_triples = self.getPrompt(sample['triples'])
            
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
            
            res = {'id': sample['id'], 'sent': chat_completion.choices[0].message.content.strip(), 'triples': chosen_triples}

            if stdout: 
                print(json.dumps(res))
                continue
            
            with open(self.output_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(res) + "\n")

    def getPrompt(self, triples: list[dict]) -> tuple[str, list[dict]]:
        prompt = f"""Your task is to generate varied, accurate, sentences about {self.prompt_subject} from a list of triples following these precise instructions :
1) Make the sentences natural, as they would appear in text or conversation, wikipedia, news or the internet.
2) The triples can be explicit in the sentence or implied, include all the triples provided, do not invent additional facts. 
3) If no currency or timezone are specified, do not include them.
4) Do not put names or titles between quotation marks like " or ' unless absolutely necessary.
5) Use varied turn of phrases, don't always start the sentence with the name of the main subject, vary using coreference instead of repeating the title. 
6) Vary the output size, sometimes a single sentence, sometimes a small paragraph, not more than a couple of sentences.
Triples:\n"""
        triples = self.getRandomNTriplesFrom(triples)
        prompt += self.getTextTriplesList(triples)
        return prompt, triples

    def getTextTriplesList(self, triples: list[dict]) -> str:
        res = ""
        for triple in triples:
            res += triple['rel'] + "(" + triple['sub'] + " <SEP> " + triple['obj'] + ")\n"
        return res

    def getRandomNTriplesFrom(self, triples: list[dict], max_n: int = 9) -> list[dict]:
        """Here we do custom random BFS like graph exploration from the first triple to generate a 
        list of n connected triples. No need for node labeling as graph has no cycles."""
        res: list[dict] = []
        queue: list[dict] = []
        
        curr = triples[0] 
        res.append(curr)
        queue.append(curr)
        
        n_triples = random.randint(2, max_n)
        print("n_triples", n_triples)
        if len(triples) < n_triples: return triples     # more triples requested than available, return all triples
        
        while len(res) != n_triples:
            queue += [triple for triple in triples if ( triple['sqid'] == curr['sqid'] or curr['oqid'] == triple['sqid']) and triple not in res]
            random.shuffle(queue)
            curr = queue.pop(0)
            if curr not in res: res.append(curr)
        
        return res


def getSamplesWithoutSentencesInOutputFile(raw_triples_path: str, output_file_path: str) -> list[dict]:
    qids_in_output_file = []

    if os.path.exists(output_file_path):
        with open(output_file_path) as f:   # get all root qids which have been verbalized
            qids_in_output_file = [json.loads(line)['id'].split("_")[-1] for line in f]
            
    raw_triples = []
    with open(raw_triples_path) as f:   # get all triples in raw_triples
        raw_triples = [json.loads(line) for line in f]
    
    # only keep those that haven't been verbalized yet
    return [sample for sample in raw_triples if sample['triples'][0]['sqid'] not in qids_in_output_file]


def worker(i, samples, output_file_path, prompt_subject, ont_name):
    print(f"T{i} deals with", len(samples), "samples, 1st sample triple sqid", samples[0]['triples'][0]['sqid'])
    
    print()
    gen = SentenceGenerator(samples, output_file_path, prompt_subject, ont_name)
    gen.generate(stdout=False)

if __name__ == "__main__":
    
    ontology_name    = sys.argv[1]      # 'ont_2_music'
    prompt_subject   = sys.argv[2]      # 'musical works'
    raw_triples_path = sys.argv[3]      # './raw_triples/ont_2_music_triples.jsonl 
    output_file_path = sys.argv[4]      # './some_path/ont_2_music_train.jsonl'

    samples = getSamplesWithoutSentencesInOutputFile(raw_triples_path, output_file_path)

    print("Found", len(samples), "that haven't been verbalized...")

    n_threads = 4
    pool = ThreadPool(n_threads)

    for i in range(n_threads):
        pool.apply_async(worker, (i, samples[i*(len(samples)//n_threads):(i+1)*(len(samples)//n_threads)], output_file_path, prompt_subject, ontology_name))
    
    pool.close()
    pool.join()
    """
    lotr_sample = {
        "triples": [
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "director", "rpid": "P57", "obj": "Peter Jackson", "oqid": "Q4465"}, 
            {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Peter Jackson", "oqid": "Q4465"}, 
            {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Philippa Boyens", "oqid": "Q116854"}, 
            {"sub": "Philippa Boyens", "sqid": "Q116854", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Fran Walsh", "oqid": "Q116861"}, 
            {"sub": "Fran Walsh", "sqid": "Q116861", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Stephen Sinclair", "oqid": "Q3498652"}, 
            {"sub": "Stephen Sinclair", "sqid": "Q3498652", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "drama film", "oqid": "Q130232"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "fantasy film", "oqid": "Q157394"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "action film", "oqid": "Q188473"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "film based on a novel", "oqid": "Q52207399"}, 
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "based on", "rpid": "P144", "obj": "The Two Towers", "oqid": "Q332388"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Peter Jackson", "oqid": "Q4465"},
            {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "John Rhys-Davies", "oqid": "Q16455"},
            {"sub": "John Rhys-Davies", "sqid": "Q16455", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Hugo Weaving", "oqid": "Q42204"},
            {"sub": "Hugo Weaving", "sqid": "Q42204", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Orlando Bloom", "oqid": "Q44467"},
            {"sub": "Orlando Bloom", "sqid": "Q44467", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Cate Blanchett", "oqid": "Q80966"},
            {"sub": "Cate Blanchett", "sqid": "Q80966", "rel": "country of citizenship", "rpid": "P27", "obj": "United States of America", "oqid": "Q30"},
            {"sub": "Cate Blanchett", "sqid": "Q80966", "rel": "country of citizenship", "rpid": "P27", "obj": "Australia", "oqid": "Q408"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Academy Award for Best Visual Effects", "oqid": "Q393686"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Academy Award for Best Sound Editing", "oqid": "Q488645"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Hugo Award for Best Dramatic Presentation, Long Form", "oqid": "Q1056240"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "production company", "rpid": "P272", "obj": "New Line Cinema", "oqid": "Q79202"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "production company", "rpid": "P272", "obj": "WingNut Films", "oqid": "Q3569323"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "country of origin", "rpid": "P495", "obj": "United States of America", "oqid": "Q30"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "country of origin", "rpid": "P495", "obj": "New Zealand", "oqid": "Q664"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2002-12-05T00:00:00Z", "oqid": "2002-12-05T00:00:00Z"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2002-12-18T00:00:00Z", "oqid": "2002-12-18T00:00:00Z"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2003-01-16T00:00:00Z", "oqid": "2003-01-16T00:00:00Z"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Gollum", "oqid": "Q15007"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Frodo Baggins", "oqid": "Q177329"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Gandalf", "oqid": "Q177499"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Aragorn", "oqid": "Q180322"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Galadriel", "oqid": "Q204274"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Picture", "oqid": "Q102427"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Production Design", "oqid": "Q277751"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Film Editing", "oqid": "Q281939"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Visual Effects", "oqid": "Q393686"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Sound", "oqid": "Q830079"},
            {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cost", "rpid": "P2130", "obj": "94000000", "oqid": "94000000"}
        ]
    }
    
    gen = SentenceGenerator([lotr_sample], '', 'films', 'ont_1_movie')
    gen.generate(stdout=True)
    """