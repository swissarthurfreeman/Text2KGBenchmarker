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

    >>> from gen_triples import TripleGenerator
    >>> gen = TripleGenerator('../ontologies/ont_1_movie.json', [
            {"qid": "Q164963", "label": "The Lord of the Rings: The Two Towers"},
            {"qid": "Q151599", "label": "Metropolis"}
        ])
    >>> gen.generate(stdout=True)
    {"triples": [{"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "director", "rpid": "P57", "obj": "Peter Jackson", "oqid": "Q4465"}, {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Peter Jackson", "oqid": "Q4465"}, {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Philippa Boyens", "oqid": "Q116854"}, {"sub": "Philippa Boyens", "sqid": "Q116854", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Fran Walsh", "oqid": "Q116861"}, {"sub": "Fran Walsh", "sqid": "Q116861", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "screenwriter", "rpid": "P58", "obj": "Stephen Sinclair", "oqid": "Q3498652"}, {"sub": "Stephen Sinclair", "sqid": "Q3498652", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "drama film", "oqid": "Q130232"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "fantasy film", "oqid": "Q157394"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "action film", "oqid": "Q188473"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "genre", "rpid": "P136", "obj": "film based on a novel", "oqid": "Q52207399"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "based on", "rpid": "P144", "obj": "The Two Towers", "oqid": "Q332388"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Peter Jackson", "oqid": "Q4465"}, {"sub": "Peter Jackson", "sqid": "Q4465", "rel": "country of citizenship", "rpid": "P27", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "John Rhys-Davies", "oqid": "Q16455"}, {"sub": "John Rhys-Davies", "sqid": "Q16455", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Hugo Weaving", "oqid": "Q42204"}, {"sub": "Hugo Weaving", "sqid": "Q42204", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Orlando Bloom", "oqid": "Q44467"}, {"sub": "Orlando Bloom", "sqid": "Q44467", "rel": "country of citizenship", "rpid": "P27", "obj": "United Kingdom", "oqid": "Q145"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cast member", "rpid": "P161", "obj": "Cate Blanchett", "oqid": "Q80966"}, {"sub": "Cate Blanchett", "sqid": "Q80966", "rel": "country of citizenship", "rpid": "P27", "obj": "United States of America", "oqid": "Q30"}, {"sub": "Cate Blanchett", "sqid": "Q80966", "rel": "country of citizenship", "rpid": "P27", "obj": "Australia", "oqid": "Q408"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Academy Award for Best Visual Effects", "oqid": "Q393686"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Academy Award for Best Sound Editing", "oqid": "Q488645"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "award received", "rpid": "P166", "obj": "Hugo Award for Best Dramatic Presentation, Long Form", "oqid": "Q1056240"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "production company", "rpid": "P272", "obj": "New Line Cinema", "oqid": "Q79202"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "production company", "rpid": "P272", "obj": "WingNut Films", "oqid": "Q3569323"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "country of origin", "rpid": "P495", "obj": "United States of America", "oqid": "Q30"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "country of origin", "rpid": "P495", "obj": "New Zealand", "oqid": "Q664"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2002-12-05T00:00:00Z", "oqid": "2002-12-05T00:00:00Z"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2002-12-18T00:00:00Z", "oqid": "2002-12-18T00:00:00Z"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "publication date", "rpid": "P577", "obj": "2003-01-16T00:00:00Z", "oqid": "2003-01-16T00:00:00Z"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Gollum", "oqid": "Q15007"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Frodo Baggins", "oqid": "Q177329"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Gandalf", "oqid": "Q177499"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Aragorn", "oqid": "Q180322"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "characters", "rpid": "P674", "obj": "Galadriel", "oqid": "Q204274"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Picture", "oqid": "Q102427"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Production Design", "oqid": "Q277751"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Film Editing", "oqid": "Q281939"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Visual Effects", "oqid": "Q393686"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "nominated for", "rpid": "P1411", "obj": "Academy Award for Best Sound", "oqid": "Q830079"}, {"sub": "The Lord of the Rings: The Two Towers", "sqid": "Q164963", "rel": "cost", "rpid": "P2130", "obj": "94000000", "oqid": "94000000"}]}
    ...
    
    {"triples": [{"sub": "Metropolis", "sqid": "Q151599", "rel": "director", "rpid": "P57", "obj": "Fritz Lang", "oqid": "Q19504"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "United States of America", "oqid": "Q30"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "Austria", "oqid": "Q40"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "Germany", "oqid": "Q183"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "screenwriter", "rpid": "P58", "obj": "Fritz Lang", "oqid": "Q19504"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "United States of America", "oqid": "Q30"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "Austria", "oqid": "Q40"}, {"sub": "Fritz Lang", "sqid": "Q19504", "rel": "country of citizenship", "rpid": "P27", "obj": "Germany", "oqid": "Q183"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "screenwriter", "rpid": "P58", "obj": "Thea von Harbou", "oqid": "Q58866"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "genre", "rpid": "P136", "obj": "drama film", "oqid": "Q130232"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "genre", "rpid": "P136", "obj": "silent film", "oqid": "Q226730"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "genre", "rpid": "P136", "obj": "science fiction film", "oqid": "Q471839"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "genre", "rpid": "P136", "obj": "dystopian film", "oqid": "Q20443008"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "genre", "rpid": "P136", "obj": "film based on a novel", "oqid": "Q52207399"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "based on", "rpid": "P144", "obj": "Metropolis", "oqid": "Q3307458"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cast member", "rpid": "P161", "obj": "Helene Weigel", "oqid": "Q60528"}, {"sub": "Helene Weigel", "sqid": "Q60528", "rel": "country of citizenship", "rpid": "P27", "obj": "Austria", "oqid": "Q40"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cast member", "rpid": "P161", "obj": "Alfred Abel", "oqid": "Q63781"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cast member", "rpid": "P161", "obj": "Heinrich George", "oqid": "Q63816"}, {"sub": "Heinrich George", "sqid": "Q63816", "rel": "country of citizenship", "rpid": "P27", "obj": "Germany", "oqid": "Q183"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cast member", "rpid": "P161", "obj": "Rudolf Klein-Rogge", "oqid": "Q65154"}, {"sub": "Rudolf Klein-Rogge", "sqid": "Q65154", "rel": "country of citizenship", "rpid": "P27", "obj": "Germany", "oqid": "Q183"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cast member", "rpid": "P161", "obj": "Gustav Fr\u00f6hlich", "oqid": "Q65496"}, {"sub": "Gustav Fr\u00f6hlich", "sqid": "Q65496", "rel": "country of citizenship", "rpid": "P27", "obj": "Germany", "oqid": "Q183"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "production company", "rpid": "P272", "obj": "UFA", "oqid": "Q41468"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "publication date", "rpid": "P577", "obj": "1927-01-01T00:00:00Z", "oqid": "1927-01-01T00:00:00Z"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "publication date", "rpid": "P577", "obj": "1927-01-10T00:00:00Z", "oqid": "1927-01-10T00:00:00Z"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "publication date", "rpid": "P577", "obj": "1927-04-04T00:00:00Z", "oqid": "1927-04-04T00:00:00Z"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "publication date", "rpid": "P577", "obj": "1927-05-06T00:00:00Z", "oqid": "1927-05-06T00:00:00Z"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "filming location", "rpid": "P915", "obj": "Berlin", "oqid": "Q64"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "love", "oqid": "Q316"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "revolution", "oqid": "Q10931"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "technology", "oqid": "Q11016"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "working class", "oqid": "Q191159"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "main subject", "rpid": "P921", "obj": "intergenerational struggle", "oqid": "Q1502007"}, {"sub": "Metropolis", "sqid": "Q151599", "rel": "cost", "rpid": "P2130", "obj": "5300000", "oqid": "5300000"}]}
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
    