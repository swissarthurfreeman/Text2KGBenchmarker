# Utility script to generate triples for a given ontology with a root entity
# for the music ontology, the root is musical work (Q2188189), for the movie 
# ontology, it's film (Q11424). This script extracts triples from wikidata. 

from multiprocessing.pool import ThreadPool
from SPARQLWrapper import SPARQLWrapper, JSON
from pandas import json_normalize
from pprint import pprint
from time import time
from time import sleep
import json
import sys
import os

def getEntitiesOfType(ent_file_path: str, triples_path: str, qid: str, n: int) -> list[dict]:
        """retrieve list of n wikidata instances of entity qid, P31 is instance of, P279 is subclass of,
        this function will filter out triples for the which we don't have a label."""
        
        sparqlwd_caller = SPARQLWrapper(
            "https://query.wikidata.org/sparql", 
            agent='TripleSentenceGeneratorBot; (github.com/swissarthurfreeman/; arthur.freeman@unige.ch)'
        )
        sparqlwd_caller.setReturnFormat(JSON)
        q = f"""
        SELECT DISTINCT ?item ?itemLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:{qid};
                    rdfs:label ?itemLabel;
                    wdt:P1843 ?whatever;
            FILTER (lang(?itemLabel) = "en").
        }}
        LIMIT {n+1}
        """
        
        sparqlwd_caller.setQuery(q)
        raw_results = sparqlwd_caller.query().convert()['results']['bindings']
        entities = [{
            "qid": res['item']['value'].split("/")[-1], 
            "label": res['itemLabel']['value']
        } for res in raw_results]
        
        qids = []
        with open(triples_path) as f:
            qids = [json.loads(line)['triples'][0]['sqid'] for line in f] 
        
        entities = [ent for ent in entities if ent['qid'] not in qids]
        
        if os.path.exists(ent_file_path):
            with open(ent_file_path, "r") as f:
                qids = [json.loads(line)['qid'] for line in f]
        
        entities = [ent for ent in entities if ent['qid'] not in qids]
        
        with open(ent_file_path, "a") as f:
            for entity in entities:
                if len(entity['label']) < 70:
                    f.write(json.dumps(entity) + "\n")
        return entities
    

def list_to_dict(dicts: list[dict], key: str) -> dict[dict[str, str]]:
    """convert list [{key :..., key1 :...}, ...] to dict {key : {key : ..., key1: ...}, key1: ...}"""
    res = {}
    for dic in dicts:
        res[dic[key]] = dic
    return res

class WKProperty:
    def __init__(self, label: str, pid: str, domain_qid: str, range_qid: str) -> None:
        self.label = label
        self.pid = pid
        self.domain_qid = domain_qid
        self.range_qid = range_qid 

class WKEntity:
    def __init__(self, label: str, qid: str) -> None:
        self.label = label
        self.qid = qid


class TripleGenerator:
    def __init__(self, user_agent: str, ontology_path: str, entities: list[dict]):
        self.user_agent = user_agent
        self.sparqlwd_caller = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
        self.sparqlwd_caller.setReturnFormat(JSON)
        self.ontology_path = ontology_path

        with open(self.ontology_path) as ont_f:
            ontology = json.load(ont_f)
            self.concepts: dict[WKEntity] = {e['qid']: WKEntity(e['label'], e['qid']) for e in ontology['concepts']}
            self.relations: dict[WKProperty] = { e['pid']: WKProperty(e['label'], e['pid'], e['domain'], e['range']) for e in ontology['relations'] }

        self.root_entities: list[dict] = entities
        
    def generate(self, thread=0) -> None:
        """Generate n triple sets for type_qid qid class instance."""
        count = 0
        
        while len(self.root_entities) > 0:
            #print("T", thread, len(self.root_entities), "entities to go")
            entity = self.root_entities.pop(0)
            #print("T", thread, "Query triples of", json.dumps(entity))
            # if a wikidata id, don't take the entity
            if entity['label'].split("Q")[-1].isdigit():
                print("continue, is digit")
                continue
            
            triples = { 'triples': self.getTriplesOfEntity(entity) }
            # if entity has some relevant triples, keep it
            if len(triples['triples']) == 0:
                print("continue, no triples")
                continue
            with open("./" + self.ontology_path.split("/")[-1] + "l", "a") as f:
                print("T", thread, " writing at", time() / 60, "minutes, count", count, "left", len(self.root_entities))
                f.write(json.dumps(triples) + "\n")
                count += 1
            
            if count % 50 == 0:
                print("T", thread, "goes to sleep.")
                sleep(60)       # sleep 5 minutes, then continue to avoid overloading request wise
        print("T", thread, "is done.")
            
    def getTriplesOfEntity(self, entity: dict) -> list[dict]:
        """retrieve all triples with qid as subject"""
        #print("Get triples of", json.dumps(entity))
        triples: list[dict] = []
        
        for pid in self.relations.keys():
            # select all objects from in prop_with_pid(entity, object), max 5, (for instance, cast member(movie, human), 5 is enough)
            q = f"""
            SELECT ?object ?objectLabel WHERE {{
                wd:{entity['qid']} wdt:{pid} ?object.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """
            self.sparqlwd_caller.setQuery(q)
            raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
            
            for obj in raw_results:
                # only keep triples that follow the ontology, award_received(film, award), entity needs to be
                # a film or a subclass of film, and object an award or subclass of award
                if self.followsOntology(entity, self.relations[pid], obj['object']['value'].split("/")[-1]):
                    triple = {
                        'sub': entity['label'],
                        'sqid': entity['qid'],
                        'rel': self.relations[pid].label,
                        'rpid': pid,
                        'obj': obj['objectLabel']['value'],
                        'oqid': obj['object']['value'].split("/")[-1]
                    }
                    triples.append(triple)
                    
                    # for example, composer(musical work, human), voice type(human, human voice), age(human, literal)
                    # cast member(film, human), country of citizenship(human, country), date of birth(human, country)
                    # note that we do not deal with recursivity, this is purely based on the DOMAIN of properties. 
                    if self.isInstanceOfDomainOfAProperty(obj['object']['value'].split("/")[-1]):
                        # get all triples relevant from the ontology going from this object
                        triples += self.getTriplesOfEntity({
                            "qid": obj['object']['value'].split("/")[-1],
                            "label": obj['objectLabel']['value']
                        })
        return triples
    
    def followsOntology(self, entity: dict, relation: WKProperty, obj_qid): 
        # here we have to check that the relation respects the range and domain, otherwise we don't keep it.
        q = f"""
        SELECT ?class WHERE {{
            wd:{entity['qid']} wdt:P31 ?x.
            ?x wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_subj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        # check that subject is instance of the class of the domain
        if relation.domain_qid not in classes_of_subj:
            return False
        
        if relation.range_qid == "": 
            return True
        
        q = f"""
        SELECT ?class WHERE {{
            wd:{obj_qid} wdt:P31 ?x.
            ?x wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_obj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        # check that object is instance of the class of the range
        if relation.range_qid not in classes_of_obj:
            return False
    
        return True
        
    def isInstanceOfDomainOfAProperty(self, obj_qid: str) -> bool:
        q = f"""
        SELECT ?class WHERE {{
            wd:{obj_qid} wdt:P31 ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_obj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        for relation in self.relations:
            if self.relations[relation].domain_qid in classes_of_obj:
                return True
        return False

def worker(i, onto_path, n_threads, entities):
    generator = TripleGenerator(
            'TripGen/3.0 (https://github.com/swissarthurfreeman/; arthur.freeman@unige.ch)',
            onto_path,
            entities=entities[i*(len(entities)//n_threads):(i+1)*(len(entities)//n_threads)]
        )
    generator.generate(i)

def removeDuplicates(triples: list[dict]) -> list[dict]:
    res = []
    for triple in triples:
        if triple not in res:
            res.append(triple)
    return res


def fold_triples(path: str) -> None:
    res: dict[str, list[dict]] = {}
    with open(path) as f:
        data = [json.loads(line) for line in f]
        for line in data:
            res[line['triples'][0]['sqid']] = removeDuplicates(line['triples'])
    
    with open(path + "_clean", "a") as f:
        for key in res:
            f.write(json.dumps({ 'id': "ont_1_movie_train_" + key, 'triples': res[key] }) + "\n")

if __name__ == "__main__":
    onto_name = sys.argv[1]         #'ont_5_military'
    root_qids = sys.argv[2:]        #['Q5', 'Q1184840', 'Q18643213', 'Q2008856', 'Q17149090']
    print(root_qids)
    # for sports, Q5 Q27020041 Q4438121
    onto_path = f'../ontologies/{onto_name}.json'
    
    # python3 gen_triples.py ont_5_military Q5 Q1184840 Q18643213 Q2008856 Q17149090
    # python3 gen_triples.py ont_3_sport Q5 Q27020041 Q4438121
    # python3 gen_triples.py ont_6_computer Q7397 Q166142 Q55990535
    # python3 gen_triples.py ont_7_space Q5 Q2488 Q3863 Q5916 Q40218
    # python3 gen_triples.py ont_9_nature Q16521 Q8502 Q355304 Q12323 Q15091377 Q7432
    #root_qid = 'Q482994'
    #root_ent_path = f"{onto_name}_root_entities_{root_qid}.jsonl"
    #getEntitiesOfType(root_ent_path, onto_name + ".jsonl", root_qid, 15_000)
    #exit(0)
    
    start = time()
    
    n_threads = 8
    pool = ThreadPool(n_threads)
    
    qids = []   # root qids for which triples have already been fetched
    with open(f"{onto_name}.jsonl") as f:
        qids = [json.loads(line)['triples'][0]['sqid'] for line in f] 
            
    entities = []   # entities for the which triples haven't yet been fetched
    for root_qid in root_qids:
        with open(f"{onto_name}_root_entities_{root_qid}.jsonl") as f:
            for line in f:
                ent = json.loads(line)
                if ent['qid'] not in qids:
                    entities.append(ent)
                
    print("Querying...")
    #entities = entities[:len(entities) - (len(entities) % n_threads)]
    print(len(entities))
    #worker(0, 1, entities)
    for i in range(n_threads):
        pool.apply_async(worker, (i, onto_path, n_threads, entities))
    
    pool.close()
    pool.join()
    
    end = time() - start
    print("This took", end / 60, "minutes for", len(entities), "samples")
    
    
    #fold_triples("./ont_1_movie.jsonl")
    