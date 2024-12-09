## Principle

Idea is to generate a dataset of all properties, to train our model to learn how to extract
arbitrary properties using attention. 

Given the following,

`(property, triples conforming to property)`, we can generate, via GPT, aligned sentences
from the triples conforming to the property. The idea is then to train our seq2seq model by 
providing it as input,


`(property, sentence)` and making it generate `(triples conforming to property)`. 

The goal, is that given a new, never before seen sentence and a new, never before seen
property, we want our model to be able to generate the triples conforming to the property.

This, at first glance seems extremely challenging, and as far as we know has never been
done before. 


The intuition behind providing `property` as input is to force attention between the 
sentence and the property, and force the model to look for bits of the sentence conforming
to the property using it's inherent knowledge.

The main limitation of current fact extraction approaches is that they simply take 
a bunch of `(sentence, triples)` pairs and ask the model to reproduce the triples, so
the model only learns to extract triples from properties it's *already seen*, it has
no idea how to extract triples conforming to new properties it's never seen before. 

## Data

There's a properties json file that wikidata provides with a list of all wikidata properties.
There's a bout 12'000 of them, a lot of them are ID related properties. 

One such entry is, 

```json
{
    "datatype": "wikibase-item",
    "id": "P54",
    "label": "member of sports team",
    "description": "sports teams or clubs that the subject represents or represented",
    "aliases": [
        "member of team",
        "team",
        "played for",
        "teams played for",
        "of team",
        "club played for",
        "player of",
        "part of team",
        "team played for",
        "plays for",
        "sport team"
    ],
    "example": [
        17515,
        462839
    ],
    "types": [
        "for items about people",
        "Wikidata qualifier"
    ]
},
```

For the `datatype` key, wikidata has the values `quantity, wikibase-item, external-id, wikibase-property`, the ones that 
actually interests us are `wikibase-item, quantity` and `external-id` instances, with the latter being unfortunately 
over represented within the dataset, it represents identifiers from external sources to wikidata, for example, 
a Taiwanese school code, an Estonian street code, or a ruchess.ru Russian chess player ID, there are 9153 such external
properties. When it comes to `quantity`, there are 515 instances, and they represent things like radial velocity of
astronomical objects or GPD growth rate, any *property whose value is a quantity*.

The most interesting property is `wikibase-item` which has instances such as `member of`, `military designation`,
`shares border with` and such, these are the traditional properties from wikidata we're familiar with, unfortunately
there are only 1670 instances of `wikibase-item`, however, most seem to have a list of aliases and examples attached
to them, as well as a types array (which is sometimes empty). Wikidata types describe the context in the which the
property should be used, for example `member of sports team` has as type `for items about people`, literally meaning
that this property should be used on people as subjects. 


## Building our Dataset

So a dataset for the task described above would be one with the following data, 


```
property1, triples conforming to property1, sentence
property2, triples conforming to property2, sentence
...
propertyN, triples conforming to property1, sentence
```

Intiuitively, we probably should only include a single instance of a given property in the dataset, and all it's aliases
or not however, does that make sense ? We really want the model to learn attention between the property and the sentence
provided as input, it'll need a lot of data to actually do that, we only have ~1.6k `wikibase-item` properties, experience
shows this is very little data to train a fact extractor, we'd need something like 100k instances.

From the property `member of sports team`, we can already create the following properties just as simple aliases, 

```
member of team,     triples conforming to member of team,   sentence
team,               triples conforming to team,             sentence
played for,         triples conforming to played for,       sentence
teams played for,   triples conforming to teams played for, sentence
of team,            triples conforming to of team,          sentence
club played for,    triples conforming to club played for,  sentence
player of,          triples conforming to player of,        sentence
part of team,       triples conforming to part of team,     sentence
team played for,    triples conforming to team played for,  sentence
plays for,          triples conforming to plays for,        sentence
sport team          triples conforming to sport team,       sentence
```

the actual triples provided could always be the same, simply replacing the relation label by the alias, for example, 
with the example `Q17515`, which corresponds to Diego Maradona, we have the triples, 

```
Diego Maradona, member of sports team, Newell's Old Boys
Diego Maradona, member of sports team, Argentino Juniors
Diego Maradona, member of sports team, SSC Napoli
Diego Maradona, member of sports team, Sevilla FC
Diego Maradona, member of sports team, FC Barcelona
Diego Maradona, member of sports team, Boca Juniors
Diego Maradona, member of sports team, Argentina men's national association football team
```

From this we can generate a simple sentence via GPT, something like,

```
sentence = "Diego Maradona was a member of several renowned football teams, including 
Newell's Old Boys, Argentinos Juniors, SSC Napoli, Sevilla FC, FC Barcelona, Boca
Juniors, and the Argentina men's national football team."
```

and then we can use the **same sentence** but with different triples, replacing the relation
label by an alias, 

```
played for triples:
---------------------
Diego Maradona, played for, Newell's Old Boys
Diego Maradona, played for, Argentino Juniors

...
Diego Maradona, played for, Argentina men's national association football team

part of team triples:
---------------------
Diego Maradona, part of team, Newell's Old Boys
Diego Maradona, part of team, Argentino Juniors
...
Diego Maradona, part of team, Argentina men's national association football team
...
```

This would yield the samples,


```
(played for,            played for triples,             sentence)
(part of team,          part of team triples,           sentence)
(member of sports team, member of sports team triples,  sentence)
(club played for,       club played for triples,        sentence)
```

So from the same sentence, we have to generate similar, but different triples. 
`Played for` intuitively has a domain of human and range of team, hopefully the 
model has enough language knowledge in it's parameters to be able to correctly
extract the triples based on this, hopefully the property name and sentence
are sufficiently informative for the model to extract stuff correctly. 

This approach however might only increase the number of samples by 3
if we're lucky, so ~4.8K samples. Which still isn't great, it's already
something though, and we can try running for multiple epochs, with high
dropout and see how well the model performs. What we're mostly interested
in is *is the model able to learn anything?* e.g. will performance on a held
out validation set that's not provided to the model increase ? We could take
about 15% of our 1.6K `wikibase-item` properties which we hold out from our  
dataset, including all their labels, and try to see if the model is able to 
increase fact extraction performance on it.  

## Domain, Range Considerations

Note that properties generally have domain and range constraints. However,
a property like `member of` has 20 value and range constraints which are possible. 
How to deal with this is not clear, though it would make sense to provide as
input property `human member of sports team`, and this would clearly be a different
property from `mascot member of sports team` but it's unclear how to build such
a dataset of properties. We'd have to do combinatorics of all possible 
`domain property range` combinations for a given `property` and then see if 
triples for that exist (most of the time, they won't), and retrieve some if they do. 

This doesn't sound totally unreasonable, though challenging and it would allow
us to generate more properties from a single one, for example, just from
`member of sports team` we can have, 

```
human, member of sports team, sports club
mascot, member of sports team, sports club
```

but for `member of` we can have, 

```
organization, member of, organization
human, member of, advocacy group
human, member of, synagogue
human, member of, subculture
human, member of, social movement
```

anything that has triples conforming to the property.


## Ontology Conformance, Range and Domain

Currently, ontology conformance doesn't check if range and domain of ontology are
respected. See fatso the cat example.