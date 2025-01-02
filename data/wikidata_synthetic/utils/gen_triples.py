################################################################
# Utility script to generate triples for a given ontology from #
# root triple files. For the music ontology, the root is       # 
# musical work (Q2188189), for the movie one, it's film        #
# (Q11424). This script extracts triples from wikidata.        #
# python3 gen_triples.py ont_3_sport Q5 Q27020041 Q4438121     #
################################################################

import os
import sys
import json
import glob
from time import time
from time import sleep
from multiprocessing.pool import ThreadPool
from SPARQLWrapper import SPARQLWrapper, JSON


class TripleGenerator:
    """Helper class to generate triples for an ontology from a list of root entities.
    The following example takes a minute or two due to they having a lot of triples in
    their respective wikidata pages. In practice, code is multithreaded (see __main__ 
    block below the class), and ran overnight on clusters. Generating all triples 
    takes multiple days.   

    >>> gen = TripleGenerator('../ontologies/ont_1_movie.json', [
            {"qid": "Q164963", "label": "The Lord of the Rings: The Two Towers"},
            {"qid": "Q151599", "label": "Metropolis"}
        ])
    >>> gen.generate(stdout=True)
    {"triples": [{"sub": "The Lord of the Rings: The Two Towers", "rel": "director", "obj": "Peter Jackson"}, {"sub": "The Lord of the Rings: The Two Towers", "rel": "screenwriter", "obj": "Peter Jackson"},
    ...
    {"triples": [{"sub": "Metropolis", "rel": "director", "obj": "Fritz Lang"}, {"sub": "Metropolis", "rel": "screenwriter", "obj": "Fritz Lang"}, {"sub": "Metropolis", "rel": "screenwriter", "obj": "Thea von Harbou"}
    ...
    """
    def __init__(self, ontology_path: str, entities: list[dict]):
        self.sparqlwd_caller = SPARQLWrapper("https://query.wikidata.org/sparql", agent='TripGen/3.0 (https://github.com/swissarthurfreeman/; arthur.freeman@unige.ch)')
        self.sparqlwd_caller.setReturnFormat(JSON)
        self.ontology_path = ontology_path

        with open(self.ontology_path) as ont_f:
            ontology = json.load(ont_f)
            self.concepts: dict[dict] = {
                e['qid']: {'label': e['label'], 'qid': e['qid']} 
                for e in ontology['concepts']
            }
            self.relations: dict[dict] = { 
                e['pid']: { 'label': e['label'], 'pid': e['pid'], 'domain_qid': e['domain'], 'range_qid': e['range'] } 
                for e in ontology['relations'] 
            }

        self.root_entities: list[dict] = entities
        
    def generate(self, thread=0, stdout: bool = False) -> None:
        """Generate n triple sets for type_qid qid class instance, write to ont_name_triples.jsonl file
        in raw_triples folder if stdout is False (default), otherwise print to stdout."""
        count = 0
        
        while len(self.root_entities) > 0:
            entity = self.root_entities.pop(0)
            
            if entity['label'].split("Q")[-1].isdigit():    # if label is a wikidata id, don't take the entity
                print("continue, is digit")
                continue
            
            triples = self.getTriplesOfEntity(entity)
            
            if len(triples) == 0:    # if entity has no relevant triples, continue
                print(f"Continue, no triples for {entity['qid']}...")
                continue
            
            if not stdout:
                with open("./raw_triples/" + self.ontology_path.split("/")[-1].replace('.json', "_triples.jsonl"), "a") as f:
                    print("T", thread, "writing at", time() / 60, "minutes, did", count, "entities left :", len(self.root_entities))
                    f.write(json.dumps({ 'triples': triples }) + "\n")
                    count += 1
            else:
                print(json.dumps({'triples': triples}))
            
            if count > 0 and count % 50 == 0:
                print("T", thread, "goes to sleep.")
                sleep(60)       # sleep a minute, then continue to avoid overloading request wise
        print("T", thread, "is done.")
    
    def getTriplesOfEntity(self, entity: dict) -> list[dict]:
        """Retrieve all triples respecting ontology with entity as root subject"""
        triples: list[dict] = []
        
        for pid in self.relations:
            # select all objects in prop_with_pid(entity, object), max 5, (for instance, cast_member(movie, human), 5 is enough)
            q = f"""
            SELECT ?object ?objectLabel WHERE {{
                wd:{entity['qid']} wdt:{pid} ?object.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """
            self.sparqlwd_caller.setQuery(q)

            # sparql call result is binding object {'object': {'type': ..., 'value': 'http://wikidata.org/entity/SOME_QID'}, 
            # 'objectLabel': {'xml:lang': 'LANG', 'type': ..., 'value': 'Literal Value'}}
            bindings: list[dict] = [{
                'objectQid': binding['object']['value'].split("/")[-1], 
                'objectLabel': binding['objectLabel']['value'],
                'subjectLabel': entity['label'],
                'subjectQid': entity['qid'],
                'relationLabel': self.relations[pid]['label'],
                'relationPid': pid,
            }  for binding in self.sparqlwd_caller.query().convert()['results']['bindings']]

            for obj in bindings:        # keep triples following ontology, if award_received(film, award), then subject needs to be film/subclassof film, object an award/subclass of award
                if self.followsOntology(entity['qid'], self.relations[pid], obj['objectQid']):
                    triples.append({'sub': entity['label'], 'sqid': obj['subjectQid'], 'rel': self.relations[pid]['label'], 'rpid': obj['relationPid'], 'obj': obj['objectLabel'], 'oqid': obj['objectQid']})
                    if self.isInstanceOfDomainOfAProperty(obj['objectQid']):    # for example, cast_member(film, human), nationality(human, country)
                        triples += self.getTriplesOfEntity({                    # get relevant ontology triples starting from this object
                            "qid": obj['objectQid'],
                            "label": obj['objectLabel']
                        })
        return triples
    
    def followsOntology(self, sub_qid: str, relation: dict, obj_qid: str) -> bool: 
        # here we have to check that the relation respects the range and domain, otherwise we don't keep it.
        q = f"""
        SELECT ?class WHERE {{
            wd:{sub_qid} wdt:P31/wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_subj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        if relation['domain_qid'] not in classes_of_subj: return False # check that subject is instance of the class of the domain
        if relation['range_qid'] == "": return True
        
        q = f"""
        SELECT ?class WHERE {{
            wd:{obj_qid} wdt:P31/wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_obj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        return relation['range_qid'] in classes_of_obj     # check that object is instance of the class of the range
        
    def isInstanceOfDomainOfAProperty(self, obj_qid: str) -> bool:
        q = f"""
        SELECT ?class WHERE {{
            wd:{obj_qid} wdt:P31/wdt:P279* ?class.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_obj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        for relation in self.relations:
            if self.relations[relation]['domain_qid'] in classes_of_obj:
                return True
        return False


def getEntitiesOfType(ent_file_path: str, pids: list[str], qid: str, n: int) -> list[dict]:
        """Retrieve list of `n` instances of entity qid with a value for every property in `pids` filter out triples without label."""
        
        sparqlwd_caller = SPARQLWrapper(
            "https://query.wikidata.org/sparql", 
            agent='TripleSentenceGeneratorBot; (github.com/swissarthurfreeman/; arthur.freeman@unige.ch)'
        )

        sparqlwd_caller.setReturnFormat(JSON)
        
        q = f"""SELECT DISTINCT ?item ?itemLabel
        WHERE {{
            ?item wdt:P31/wdt:P279* wd:{qid};
                    rdfs:label ?itemLabel;"""
        
        for pid in pids: q += f"wdt:{pid} ?value_for_{pid};"
        
        q+= f"""FILTER (lang(?itemLabel) = "en").}} LIMIT {n}"""
        
        sparqlwd_caller.setQuery(q)
        raw_results = sparqlwd_caller.query().convert()['results']['bindings']
        
        entities = [{
            "qid": res['item']['value'].split("/")[-1], 
            "label": res['itemLabel']['value']
        } for res in raw_results]
        
        if os.path.exists(ent_file_path):
            print(ent_file_path, "root entities file already exists, aborting to avoid duplicates...")

        with open(ent_file_path, "a") as f:
            for entity in entities:
                if len(entity['label']) < 70 and not entity['label'].split("Q")[-1].isdigit():    # if label ain't too long and isn't an id
                    f.write(json.dumps(entity) + "\n")                                            # take the entity
        return entities


def getUnprocessedRootEntities(onto_name: str) -> list[dict]:
    root_qids_with_triples: list[dict] = []
    with open(f"./raw_triples/{onto_name}_triples.jsonl") as f:
        root_qids_with_triples = [json.loads(line)['triples'][0]['sqid'] for line in f] 
    
    root_qids_without_triples: list[dict] = []
    root_qids_files = glob.glob(f"./root_entities/{onto_name}_root_entities_**.jsonl")

    for root_qid_file in root_qids_files: 
        with open(root_qid_file) as f:
            for line in f:
                ent = json.loads(line)
                if ent['qid'] not in root_qids_with_triples:    # if qid hasn't doesn't have a triples list
                    root_qids_without_triples.append(ent)

    return root_qids_without_triples


def worker(i, onto_path, n_threads, entities):
    generator = TripleGenerator(
            onto_path,
            entities=entities[i*(len(entities)//n_threads):(i+1)*(len(entities)//n_threads)]
        )
    generator.generate(i)


if __name__ == "__main__":
    onto_name = sys.argv[1]         # 'ont_5_military'
    root_qids = sys.argv[2:]        # ['Q5', 'Q1184840', 'Q18643213', 'Q2008856', 'Q17149090']
    
    assert len(root_qids) > 0, "No root entities specified, aborting..."

    start = time()
    
    n_threads = 5
    pool = ThreadPool(n_threads)
    
    root_entities_without_triples = getUnprocessedRootEntities(onto_name)
    print("Retrieved", len(root_entities_without_triples), "root entities to get triples of.")

    for i in range(n_threads):
        pool.apply_async(worker, (i, f'../ontologies/{onto_name}.json', n_threads, root_entities_without_triples))
    
    pool.close()
    pool.join()
    
    end = time() - start
    print("This took", end / 60, "minutes for", len(root_entities_without_triples), "samples")
    
    # python3 gen_triples.py ont_5_military Q5 Q1184840 Q18643213 Q2008856 Q17149090
    # python3 gen_triples.py ont_3_sport Q5 Q27020041 Q4438121
    # python3 gen_triples.py ont_6_computer Q7397 Q166142 Q55990535
    # python3 gen_triples.py ont_7_space Q2488 Q3863 Q5916 Q40218 Q5
    # python3 gen_triples.py ont_9_nature Q16521 Q8502 Q355304 Q12323 Q15091377 Q7432
    #root_qid = 'Q482994'
    #root_ent_path = f"{onto_name}_root_entities_{root_qid}.jsonl"
    #getEntitiesOfType(root_ent_path, onto_name + ".jsonl", root_qid, 15_000)
    #exit(0)
    