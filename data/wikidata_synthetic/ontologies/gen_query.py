import json


with open("ont_1_movie.json") as f:
    ont = json.load(f)
    
    query = "SELECT DISTINCT ?filmLabel"
    
    for rel in ont['relations']:
        rel['label'] = rel['label'].replace(" ", "_")
        query += " ?" + rel['label'] + "Label"
    
    query += " WHERE {\n"
    query += "  ?film wdt:P31/wdt:P279* wd:Q11424;\n"
    query += "      rdfs:label ?filmLabel."
    
    for rel in ont['relations']:
        query += f"  ?film wdt:{rel['pid']} ?{rel['label']}.\n"
        query += f"  ?{rel['label']} rdfs:label ?{rel['label']}Label.\n"
    
    query += f' FILTER(lang(?filmLabel) = "en")\n'
    for rel in ont['relations']:
        query += f' FILTER(lang(?{rel["label"]}Label) = "en")\n'
    
    query += "}\n"
    print(query)