sent = "Hai-Tang is a 1930 British-German drama film directed by 33 year old Richard Eichberg starring Marcel Vibert and Robert Ancelin."

sent = "This Must Be the Place” is a song by new wave band Talking Heads, released in November 1983 as the second single from its fifth album “Speaking in Tongues”"
gold = "<triplet> This Must Be the Place <subj> Talking Heads <obj> performer <subj> Speaking in Tongues <obj> part of <triplet> Talking Heads <subj> new wave <obj> genre <triplet> Speaking in Tongues <subj> Talking Heads <obj> performer"


triplets = [
    {"sub": "Hai-Tang", "rel": "cast member", "obj": "Marcel Vibert"},
    {"sub": "Hai-Tang", "rel": "cast member", "obj": "Robert Ancelin"},
    {"sub": "Hai-Tang", "rel": "director", "obj": "Richard Eichberg"},
    {"sub": "Hai-Tang", "rel": "publication date", "obj": "01 January 1930"},
    {"sub": "Richard Eichberg", "rel": "age", "obj": "33"}
]

triplets = [
    {"sub": "This Must Be the Place", "rel": "part of",   "obj": "Speaking in Tongues"},
    {"sub": "Speaking in Tongues",    "rel": "performer", "obj": "Talking Heads"},
    {"sub": "This Must Be the Place", "rel": "performer", "obj": "Talking Heads"},
    {"sub": "Talking Heads",          "rel": "genre",     "obj": "new wave"},
]


# this implementation doesn't sort the entity set, it's not correct
# for some reason, in the original dataset scripts, in NYT there's one 
# sort on relations, but on rebel dataset there's twom and in conll04 
# there are none... Perhaps the data is already pre-sorted as we want ? 
def lin_triplets(triples, sent):
    lin_triplets: str = ""
    entities = sorted({triple['sub'] for triple in triples}, key=lambda sub: sent.find(sub))

    for ent in entities:
        if len(lin_triplets) == 0:
            lin_triplets += '<triplet> ' + ent
        else:
            lin_triplets += ' <triplet> ' + ent
        relations = sorted([triple for triple in triples if triple['sub'] == ent], key=lambda trip: sent.find(trip['obj']))
        
        for rel in relations:
            lin_triplets += ' <subj> ' + rel['obj'] + ' <obj> ' + rel['rel']
        
    return lin_triplets
    
print(lin_triplets(triplets, sent))