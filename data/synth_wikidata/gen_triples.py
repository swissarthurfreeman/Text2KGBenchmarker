from SPARQLWrapper import SPARQLWrapper, JSON
from pandas import json_normalize
from pprint import pprint
import json

def list_to_dict(dicts: list[dict], key: str) -> dict[dict[str, str]]:
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
    def __init__(self, user_agent: str, ontology_path: str):
        self.user_agent = user_agent
        self.sparqlwd_caller = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
        self.sparqlwd_caller.setReturnFormat(JSON)
        self.ontology_path = ontology_path

        with open(self.ontology_path) as ont_f:
            ontology = json.load(ont_f)
            self.concepts: dict[WKEntity] = {e['qid']: WKEntity(e['label'], e['qid']) for e in ontology['concepts']}
            self.relations: dict[WKProperty] = { e['pid']: WKProperty(e['label'], e['pid'], e['domain'], e['range']) for e in ontology['relations'] }

        self.root_entities: list[dict] = []
    
    def getEntitiesOfType(self, qid: str) -> list[dict]:
        """retrieve list of wikidata instances of entity qid"""
        q = f"""
        SELECT ?item ?itemLabel WHERE {{
            ?item wdt:P31 wd:{qid}.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 10
        """
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        self.entities = [{
            "qid": res['item']['value'].split("/")[-1], 
            "label": res['itemLabel']['value']
        } for res in raw_results]
        return self.entities
    
    def getTriplesOfEntity(self, entity: dict) -> list[dict]:
        """retrieve all triples with qid as subject"""
        print("get triples of ", entity)
        triples: list[dict] = []
        
        for pid in self.relations.keys():
            # select all objects from in prop_with_pid(entity, object)
            q = f"""
            SELECT ?object ?objectLabel WHERE {{
                wd:{entity['qid']} wdt:{pid} ?object.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 10
            """
            self.sparqlwd_caller.setQuery(q)
            raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
            
            for obj in raw_results:
                # only keep triples that follow the ontology, award_received(film, award), entity needs to be
                # a film or a subclass of film, and object an award or subclass of award
                if self.followsOntology(entity, self.relations[pid], obj['object']['value'].split("/")[-1]):
                    triple = {
                        'sub': entity['label'],
                        'rel': self.relations[pid].label,
                        'obj': obj['objectLabel']['value']
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
        print("query", q)
        self.sparqlwd_caller.setQuery(q)
        raw_results = self.sparqlwd_caller.query().convert()['results']['bindings']
        classes_of_obj = [res['class']['value'].split("/")[-1] for res in raw_results]
        
        for relation in self.relations:
            if self.relations[relation].domain_qid in classes_of_obj:
                return True
        return False 
            
    
    

if __name__ == "__main__":
    generator = TripleGenerator(
        'TripleSentenceGeneratorBot/0.0 (https://github.com/swissarthurfreeman/; arthur.freeman@unige.ch)',
        '../wikidata_tekgen/ontologies/ont_1_movie.json'
    )
    
    # Q11424 is film entity, root entity is the main entity of interest
    # of the ontology, like for the UN it would be a resolution
    intouchables = generator.getEntitiesOfType("Q11424")[2]
    pprint(intouchables)
    
    pprint(generator.getTriplesOfEntity(intouchables))
    