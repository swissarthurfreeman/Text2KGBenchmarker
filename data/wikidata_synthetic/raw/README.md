# Raw Wikidata Triples

All the files here are the extracted, **raw, unprocessed triples** for every ontology
of wikidata (initially movie and music ones). Every entry contains all triples
relevant to the ontology for a given film, musical work etc, depending on the ontology.

The `../train` contains sentences generated from *a subset* of these triples, as there 
are often too many to generate a short paragraph, this pre-processing step is purely 
heuristic, and can be done a variety of ways. We give the user the freedom to choose
his own prompt and triple selection strategy by letting him modify the `../gen_sent.py`
triples as he likes. 