"""Matched recall/derivation probes, built to separate MECHANISM from TOPIC.

R1 found that expert selection separates knowledge prompts from math prompts
against a proper null. That result cannot distinguish two explanations:

  mechanism   the router routes differently when the model RETRIEVES a
              memorised fact than when it COMPUTES an answer
  topic       the router routes differently because "Canberra" and "divisors"
              are different subject matter

Only the first is worth anything. A meter that fires on vocabulary is useless
on free text, and every W5-family probe set so far confounds the two: the
the KNOWLEDGE and REASONING probe lists share almost no vocabulary.

So each fact here generates BOTH classes from the SAME entities:

  recall   "The Treaty of Westphalia was signed in"                  -> 1648
  derive   "The Treaty of Westphalia was signed in 1648. Three years
            earlier was"                                             -> 1645

The derive item STATES the fact, so nothing needs retrieving — the answer is
computed from a premise sitting in the context — while the tokens, register and
subject matter stay put. A router signal that separates these is measuring
mechanism. One that does not was measuring topic, and R1's effect dissolves.

DERIVE ANSWERS ARE NEVER COPYABLE FROM THE PREMISE. "The first letter of
Canberra is C" would be a lookup into the context, not a derivation, and the
model could solve it by attending to the span — which is a third mechanism,
distinct from both the ones being contrasted. Letter COUNTS and year ARITHMETIC
both require computation over the premise rather than retrieval of it.

Every fact also carries several PARAPHRASES under one `fact_id`. That is what
makes rung 3 of the readability ladder testable: is a fact's routing signature
stable across surface forms and distinct across facts? If yes, the routing
pattern is a readable ADDRESS for that fact. Paraphrases are hand-written
rather than templated, because a templated paraphrase set would measure
template identity rather than the fact.

The suite deliberately reuses facts from that KNOWLEDGE list
where possible: those are already validated to survive `validate()` on both
checkpoints (46/50 kept on qwen), so probe attrition here is predictable
instead of discovered on the GPU.

Stdlib only, no model, no GPU.

Usage:
  python3 -m knowledge.probes --stats
  python3 -m knowledge.probes --dump probes.json
"""
import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'records'

# kind drives which derivation templates apply:
#   'year'   answer is a bare number -> arithmetic offsets
#   'entity' answer is a word        -> letter counts
# `value` is the numeric answer for year facts; `subject` is a single word from
# the stem used as a second, independent letter-count target so entity facts
# get two derive items without repeating the same computation.
F = [
    # ---- geography -------------------------------------------------------
    dict(id='geo.au.capital', domain='geography', kind='entity',
         answer='Canberra', subject='Australia', stems=[
             "The capital city of Australia is",
             "Australia's seat of government is the city of",
             "The Australian city that houses Parliament House is"]),
    dict(id='geo.ca.capital', domain='geography', kind='entity',
         answer='Ottawa', subject='Canada', stems=[
             "The capital of Canada is",
             "Canada's federal capital city is",
             "The city where Canada's Parliament sits is"]),
    dict(id='geo.mn.capital', domain='geography', kind='entity',
         answer='Ulaanbaatar', subject='Mongolia', stems=[
             "The capital of Mongolia is",
             "Mongolia's largest city and capital is",
             "The seat of Mongolia's government is"]),
    dict(id='geo.no.capital', domain='geography', kind='entity',
         answer='Oslo', subject='Norway', stems=[
             "The capital of Norway is",
             "Norway's capital city is",
             "The Norwegian city where the government sits is"]),
    dict(id='geo.pt.capital', domain='geography', kind='entity',
         answer='Lisbon', subject='Portugal', stems=[
             "The capital of Portugal is",
             "Portugal's capital city is",
             "The Portuguese capital, on the Tagus, is"]),
    dict(id='geo.vn.capital', domain='geography', kind='entity',
         answer='Hanoi', subject='Vietnam', stems=[
             "The capital of Vietnam is",
             "Vietnam's capital city is",
             "The northern Vietnamese city that is its capital is"]),
    dict(id='geo.budapest.river', domain='geography', kind='entity',
         answer='Danube', subject='Budapest', stems=[
             "The river that flows through Budapest is the",
             "Budapest is built on both banks of the river",
             "The great river dividing Buda from Pest is the"]),
    dict(id='geo.longest.river', domain='geography', kind='entity',
         answer='Nile', subject='Africa', stems=[
             "The longest river in the world is the",
             "The river flowing north through Egypt and Sudan is the",
             "Africa's longest river is the"]),
    dict(id='geo.largest.desert', domain='geography', kind='entity',
         answer='Sahara', subject='Africa', stems=[
             "The largest hot desert in the world is the",
             "The vast desert covering northern Africa is the",
             "Africa's great northern desert is the"]),
    dict(id='geo.deepest.trench', domain='geography', kind='entity',
         answer='Mariana', subject='Pacific', stems=[
             "The deepest ocean trench is the",
             "The deepest point in the Pacific lies in the",
             "The trench containing Challenger Deep is the"]),

    # ---- chemistry -------------------------------------------------------
    dict(id='chem.fe.symbol', domain='chemistry', kind='entity',
         answer='iron', subject='ferrum', stems=[
             "The element with atomic number 26 is",
             "The metal whose chemical symbol is Fe is",
             "The most common element by mass in Earth's core is"]),
    dict(id='chem.au.name', domain='chemistry', kind='entity',
         answer='gold', subject='aurum', stems=[
             "The element with the chemical symbol Au is",
             "The precious yellow metal with atomic number 79 is",
             "The metal whose symbol comes from 'aurum' is"]),
    dict(id='chem.na.name', domain='chemistry', kind='entity',
         answer='sodium', subject='natrium', stems=[
             "The element with the chemical symbol Na is",
             "The alkali metal with atomic number 11 is",
             "The metal whose symbol comes from 'natrium' is"]),
    dict(id='chem.k.name', domain='chemistry', kind='entity',
         answer='potassium', subject='kalium', stems=[
             "The element with the chemical symbol K is",
             "The alkali metal with atomic number 19 is",
             "The element whose symbol comes from 'kalium' is"]),
    dict(id='chem.salt.formula', domain='chemistry', kind='entity',
         answer='chloride', subject='sodium', stems=[
             "In table salt, sodium is paired with",
             "The compound NaCl is sodium",
             "Table salt is chemically sodium"]),
    dict(id='chem.plants.gas', domain='chemistry', kind='entity',
         answer='carbon', subject='photosynthesis', stems=[
             "The gas plants absorb during photosynthesis is",
             "Plants take in from the air the gas known as",
             "The atmospheric gas fixed by photosynthesis is"]),
    dict(id='chem.hardest', domain='chemistry', kind='entity',
         answer='diamond', subject='carbon', stems=[
             "The hardest natural substance is",
             "The carbon allotrope that tops the Mohs scale is",
             "The hardest naturally occurring mineral is"]),

    # ---- astronomy -------------------------------------------------------
    dict(id='astro.closest.planet', domain='astronomy', kind='entity',
         answer='Mercury', subject='Sun', stems=[
             "The planet closest to the Sun is",
             "The innermost planet of the solar system is",
             "The smallest planet, nearest the Sun, is"]),
    dict(id='astro.largest.planet', domain='astronomy', kind='entity',
         answer='Jupiter', subject='solar', stems=[
             "The largest planet in the solar system is",
             "The gas giant with the Great Red Spot is",
             "The most massive planet orbiting the Sun is"]),
    dict(id='astro.saturn.moon', domain='astronomy', kind='entity',
         answer='Titan', subject='Saturn', stems=[
             "The largest moon of Saturn is",
             "Saturn's moon with a thick nitrogen atmosphere is",
             "The second-largest moon in the solar system, orbiting Saturn, is"]),

    # ---- literature ------------------------------------------------------
    dict(id='lit.leopard.author', domain='literature', kind='entity',
         answer='Tomasi', subject='Leopard', stems=[
             "The novel 'The Leopard' was written by Giuseppe",
             "'Il Gattopardo' was the work of Giuseppe",
             "The Sicilian author of 'The Leopard' was Giuseppe"]),
    dict(id='lit.solitude.author', domain='literature', kind='entity',
         answer='Marquez', subject='Colombia', stems=[
             "The author of 'One Hundred Years of Solitude' is Gabriel Garcia",
             "'Cien anos de soledad' was written by Gabriel Garcia",
             "The Colombian Nobel laureate who wrote of Macondo is Gabriel Garcia"]),
    dict(id='lit.1984.author', domain='literature', kind='entity',
         answer='Orwell', subject='England', stems=[
             "The author of 'Nineteen Eighty-Four' was George",
             "'Animal Farm' and '1984' were both written by George",
             "The English writer who created Big Brother was George"]),
    dict(id='lit.pride.author', domain='literature', kind='entity',
         answer='Austen', subject='England', stems=[
             "The author of 'Pride and Prejudice' was Jane",
             "'Emma' and 'Persuasion' were written by Jane",
             "The English novelist who created Elizabeth Bennet was Jane"]),
    dict(id='lit.hamlet.author', domain='literature', kind='entity',
         answer='Shakespeare', subject='Hamlet', stems=[
             "The playwright who wrote 'Hamlet' was William",
             "'Macbeth' and 'King Lear' were written by William",
             "The Stratford-born playwright of 'Hamlet' was William"]),

    # ---- art and music ---------------------------------------------------
    dict(id='art.meninas.painter', domain='art', kind='entity',
         answer='Velazquez', subject='Meninas', stems=[
             "The painter of 'Las Meninas' was Diego",
             "'Las Meninas' hangs in the Prado, painted by Diego",
             "The Spanish court painter of Philip IV was Diego"]),
    dict(id='art.david.sculptor', domain='art', kind='entity',
         answer='Michelangelo', subject='Florence', stems=[
             "The sculptor of the statue of David was",
             "The Florentine who carved David and painted the Sistine ceiling was",
             "The marble David in the Accademia was sculpted by"]),
    dict(id='art.rite.composer', domain='art', kind='entity',
         answer='Stravinsky', subject='Russia', stems=[
             "The composer of 'The Rite of Spring' was Igor",
             "'Le Sacre du printemps' was composed by Igor",
             "The Russian composer who scandalised Paris in 1913 was Igor"]),
    dict(id='art.ninth.composer', domain='art', kind='entity',
         answer='Beethoven', subject='Vienna', stems=[
             "The composer of the Ninth Symphony was Ludwig van",
             "The 'Ode to Joy' finale was written by Ludwig van",
             "The deaf Viennese composer of nine symphonies was Ludwig van"]),

    # ---- history: year facts (arithmetic derivations) --------------------
    dict(id='hist.westphalia', domain='history', kind='year',
         answer='1648', value=1648, subject='Westphalia', stems=[
             "The Treaty of Westphalia was signed in",
             "The peace ending the Thirty Years' War was concluded in",
             "The Westphalian settlement was agreed in the year"]),
    dict(id='hist.moon.landing', domain='history', kind='year',
         answer='1969', value=1969, subject='Apollo', stems=[
             "The first crewed Moon landing took place in",
             "Apollo 11 touched down on the Moon in",
             "Neil Armstrong first walked on the Moon in the year"]),
    dict(id='hist.ww2.end', domain='history', kind='year',
         answer='1945', value=1945, subject='war', stems=[
             "The Second World War ended in",
             "Victory in Europe and Japan both came in",
             "The war in the Pacific concluded in the year"]),
    dict(id='hist.berlin.wall', domain='history', kind='year',
         answer='1989', value=1989, subject='Berlin', stems=[
             "The Berlin Wall fell in",
             "East Germany opened its border crossings in",
             "The wall dividing Berlin came down in the year"]),
    dict(id='hist.french.revolution', domain='history', kind='year',
         answer='1789', value=1789, subject='Bastille', stems=[
             "The storming of the Bastille took place in",
             "The French Revolution began in",
             "The Bastille was stormed in the year"]),
    dict(id='hist.magna.carta', domain='history', kind='year',
         answer='1215', value=1215, subject='Runnymede', stems=[
             "Magna Carta was sealed in",
             "King John met the barons at Runnymede in",
             "The Great Charter was agreed in the year"]),
]

# Derivation offsets, applied to `value` for year facts. Small and exact so the
# model can do them in ONE step — the original validate() exists because a probe
# the model cannot solve unaided measures noise, and "17 times 23" was the
# lesson that cost. Offsets crossing a century boundary are avoided for the
# same reason.
YEAR_OFFSETS = ((3, 'earlier'), (2, 'later'))


def letters(word):
    return sum(c.isalpha() for c in word)


def build():
    """[(probe dict)] — every recall and derive item, fully labeled.

    Fields carried through to the capture so the analysis can group correctly:
      fact_id      same fact, any surface form  -> rung 3 (routing signature)
      para         which paraphrase             -> stability within a fact
      cls          'recall' | 'derive'          -> the meter's label
      matched      True if this shares vocabulary with its recall counterpart
    """
    out = []
    for f in F:
        for i, stem in enumerate(f['stems']):
            out.append({'probe_id': f"{f['id']}.r{i}", 'fact_id': f['id'],
                        'domain': f['domain'], 'cls': 'recall', 'para': i,
                        'dkind': None, 'matched': True, 'suite': 'mechanism',
                        'atype': 'num' if f['answer'][0].isdigit() else 'word',
                        'stem': stem, 'answer': ' ' + f['answer']})

        # the premise states the fact, so the derive items need no retrieval
        premise = f"{f['stems'][0]} {f['answer']}."
        ans, subj = f['answer'], f['subject']
        # operands vary per fact so the suite does not ask the same sum 35
        # times — repeated arithmetic would give every fact an identical
        # derive-side routing pattern and contaminate the fact-signature test
        n = len(out)
        a, b = 21 + (n * 7) % 40, 3 + (n * 5) % 15
        c, d = 12 + (n * 3) % 30, 2 + (n * 11) % 9
        items = []
        if f['kind'] == 'year':
            for off, direction in YEAR_OFFSETS:
                v = f['value'] - off if direction == 'earlier' else f['value'] + off
                items.append(('year',
                              f"{premise} The year {off} years {direction} was",
                              str(v)))
        else:
            items.append(('arith',
                          f"{premise} A survey mentioned {ans} {a} times and "
                          f"{subj} {b} times. The difference is", str(a - b)))
            items.append(('arith',
                          f"{premise} {ans} appears on {c} maps and {subj} on "
                          f"{d} more. {subj} appears on", str(c + d)))
            # kept deliberately, and expected to attrit: letter counting is a
            # tokenizer weakness, so validate() will drop much of it. Its value
            # is as a CONTRAST — if the meter reads arithmetic-derivation but
            # not letter-derivation, it is keyed to one kind of computation
            # rather than to "not retrieval", which is worth knowing.
            items.append(('letters',
                          f"{premise} The number of letters in the word "
                          f"'{ans}' is", str(letters(ans))))
        for j, (dkind, stem, answer) in enumerate(items):
            out.append({'probe_id': f"{f['id']}.d{j}", 'fact_id': f['id'],
                        'domain': f['domain'], 'cls': 'derive', 'para': j,
                        'dkind': dkind, 'matched': True, 'suite': 'mechanism',
                        'atype': 'num' if answer[0].isdigit() else 'word',
                        'stem': stem, 'answer': ' ' + answer})
    return out


# Unmatched derivation: shares NO vocabulary with any recall probe. This is the
# topic baseline — the quantity R1 could not separate from mechanism. If the
# meter scores matched and unmatched derivations alike, it generalizes; if it
# only works on the unmatched set, it is a topic detector wearing a disguise.
UNMATCHED = [
    ("Compute: 6 times 7 equals", " 42"),
    ("Compute: 12 plus 8 equals", " 20"),
    ("Compute: 100 minus 37 equals", " 63"),
    ("Compute: 144 divided by 12 equals", " 12"),
    ("Compute: 8 squared equals", " 64"),
    ("If 5x = 45, then x equals", " 9"),
    ("If x + 12 = 30, then x equals", " 18"),
    ("If 3x = 21, then x equals", " 7"),
    ("The sum of the first 10 positive integers is", " 55"),
    ("The next number in the sequence 2, 4, 8, 16 is", " 32"),
    ("The area of a rectangle 6 by 7 is", " 42"),
    ("The number of minutes in 3 hours is", " 180"),
    ("Half of 96 is", " 48"),
    ("Double 35 is", " 70"),
    ("The perimeter of a square with side 9 is", " 36"),
]


def build_unmatched():
    return [{'probe_id': f'unmatched.d{i}', 'fact_id': f'unmatched.{i}',
             'domain': 'control', 'cls': 'derive', 'para': 0,
             'dkind': 'arith', 'matched': False, 'suite': 'mechanism',
             'atype': 'num', 'stem': s, 'answer': a}
            for i, (s, a) in enumerate(UNMATCHED)]


def build_grounding():
    """Parametric vs contextual, with the ANSWER TOKEN HELD IDENTICAL.

    The first suite (`build`) controlled topic and vocabulary and was still
    confounded: every derivation it writes yields a NUMBER (arithmetic, letter
    counts, year offsets) while most recall answers are words, so a classifier
    scored 1.000 on class and 0.995 on "is the answer numeric" — the same
    information. Restricted to numeric answers only it fell to chance. It was
    reading the form of the next token, not the mechanism.

    Here both classes ask the SAME question and emit the SAME answer token. The
    only difference is whether the fact is available in context:

      parametric   {stem}                              -> answer, from weights
      contextual   Fact: {other paraphrase} {answer}.
                   {stem}                              -> answer, from context

    Answer type cannot confound this, because the two sides share the answer.
    The context sentence uses a DIFFERENT paraphrase than the question, so the
    contextual case is grounding rather than verbatim span-copying.

    A THIRD CLASS closes the length confound. `contextual` prompts carry a
    prepended sentence and `parametric` ones do not, so prompt length — and
    with it the answer-token POSITION — separates them for free. `distractor`
    prepends an UNRELATED fact instead: same shape, same length distribution,
    same "Fact: ..." frame, but the answer is still only in the weights.

      contextual vs distractor   the clean test. Both have context; only one
                                 contains the answer.
      parametric                 kept for reference, not for the headline.

    This is also the more useful diagnostic. "Did this token come from the
    provided context or from parametric memory" is the RAG-grounding question,
    and the parametric side is precisely the knowledge that would have to be
    externalised.
    """
    out = []
    for fi, f in enumerate(F):
        ans = f['answer']
        atype = 'num' if ans[0].isdigit() else 'word'
        # an unrelated fact, for the length-matched control below
        d = F[(fi + 7) % len(F)]
        for i, stem in enumerate(f['stems']):
            other = f['stems'][(i + 1) % len(f['stems'])]
            for cls, text in (
                    ('parametric', stem),
                    ('contextual', f"Fact: {other} {ans}. {stem}"),
                    ('distractor',
                     f"Fact: {d['stems'][i % len(d['stems'])]} "
                     f"{d['answer']}. {stem}")):
                out.append({
                    'probe_id': f"{f['id']}.{cls[:4]}{i}", 'fact_id': f['id'],
                    'domain': f['domain'], 'cls': cls, 'para': i,
                    'dkind': None, 'matched': True, 'suite': 'grounding',
                    'atype': atype, 'stem': text, 'answer': ' ' + ans})
    return out


# ---------------------------------------------------------------------------
# R3 — recall vs COMPUTATION, with the answer token identical across classes.
#
# The question R2 set out to answer and could not. Its first suite scored 1.000
# by reading "is the next token a digit"; its second suite answered a different
# question (context vs weights). Here both classes emit the SAME NUMBER, so
# neither the form nor the magnitude of the answer can leak.
#
#   retrieved   "... The Treaty of Westphalia was signed in the year"  -> 1648
#   computed    "... The year 3 years after 1645 is"                   -> 1648
#
# Same answer token, same numeric magnitude, same prefix sentence. One requires
# a fact from the weights; the other is arithmetic over an operand in the
# prompt. `value` is what makes the pairing possible: the arithmetic is
# constructed backwards from the fact's own value.
NUMERIC = [
    dict(id='y.westphalia', domain='history', value=1648, kind='year', stems=[
        "The Treaty of Westphalia was signed in the year",
        "The peace ending the Thirty Years' War was concluded in the year",
        "The Westphalian settlement was agreed in the year"]),
    dict(id='y.moon', domain='history', value=1969, kind='year', stems=[
        "The first crewed Moon landing took place in the year",
        "Apollo 11 touched down on the Moon in the year",
        "Neil Armstrong first walked on the Moon in the year"]),
    dict(id='y.ww2end', domain='history', value=1945, kind='year', stems=[
        "The Second World War ended in the year",
        "Victory in Europe and Japan both came in the year",
        "The war in the Pacific concluded in the year"]),
    dict(id='y.wall', domain='history', value=1989, kind='year', stems=[
        "The Berlin Wall fell in the year",
        "East Germany opened its border crossings in the year",
        "The wall dividing Berlin came down in the year"]),
    dict(id='y.bastille', domain='history', value=1789, kind='year', stems=[
        "The storming of the Bastille took place in the year",
        "The French Revolution began in the year",
        "The Bastille fell to the crowd in the year"]),
    dict(id='y.magna', domain='history', value=1215, kind='year', stems=[
        "Magna Carta was sealed in the year",
        "King John met the barons at Runnymede in the year",
        "The Great Charter was agreed in the year"]),
    dict(id='y.declaration', domain='history', value=1776, kind='year', stems=[
        "The American Declaration of Independence was adopted in the year",
        "The thirteen colonies declared independence in the year",
        "Independence was declared in Philadelphia in the year"]),
    dict(id='y.titanic', domain='history', value=1912, kind='year', stems=[
        "The Titanic sank in the year",
        "The Titanic struck an iceberg on her maiden voyage in the year",
        "The White Star liner Titanic was lost in the year"]),
    dict(id='y.chernobyl', domain='history', value=1986, kind='year', stems=[
        "The Chernobyl disaster occurred in the year",
        "Reactor four at Chernobyl exploded in the year",
        "The worst nuclear accident in history happened in the year"]),
    dict(id='y.flight', domain='history', value=1903, kind='year', stems=[
        "The Wright brothers made their first powered flight in the year",
        "The first sustained powered aeroplane flight was made in the year",
        "Kitty Hawk saw the first powered flight in the year"]),
    dict(id='y.fire', domain='history', value=1666, kind='year', stems=[
        "The Great Fire of London broke out in the year",
        "London burned for four days in the year",
        "The fire that began in Pudding Lane occurred in the year"]),
    dict(id='y.hastings', domain='history', value=1066, kind='year', stems=[
        "The Battle of Hastings was fought in the year",
        "William the Conqueror invaded England in the year",
        "Harold fell at Hastings in the year"]),
    dict(id='y.constantinople', domain='history', value=1453, kind='year',
         stems=[
        "Constantinople fell to the Ottomans in the year",
        "The Byzantine Empire ended with the fall of its capital in the year",
        "Mehmed II captured Constantinople in the year"]),
    dict(id='y.civilwar', domain='history', value=1865, kind='year', stems=[
        "The American Civil War ended in the year",
        "Lee surrendered at Appomattox in the year",
        "The Confederacy collapsed in the year"]),
    dict(id='y.russian', domain='history', value=1917, kind='year', stems=[
        "The Russian Revolution took place in the year",
        "The Bolsheviks seized power in Petrograd in the year",
        "The Tsar abdicated in the year"]),
    dict(id='y.versailles', domain='history', value=1919, kind='year', stems=[
        "The Treaty of Versailles was signed in the year",
        "The peace treaty ending the First World War was signed in the year",
        "The Paris Peace Conference concluded its main treaty in the year"]),
    dict(id='y.armada', domain='history', value=1588, kind='year', stems=[
        "The Spanish Armada was defeated in the year",
        "Philip II's fleet failed against England in the year",
        "The Armada sailed against England in the year"]),
    dict(id='y.olympics', domain='history', value=1896, kind='year', stems=[
        "The first modern Olympic Games were held in the year",
        "Athens hosted the first modern Olympics in the year",
        "The modern Olympic movement began its games in the year"]),
    dict(id='y.columbus', domain='history', value=1492, kind='year', stems=[
        "Columbus first reached the Americas in the year",
        "The first Columbian voyage made landfall in the year",
        "Spain's expedition west across the Atlantic arrived in the year"]),
    dict(id='y.ww1end', domain='history', value=1918, kind='year', stems=[
        "The First World War ended in the year",
        "The Armistice on the Western Front was signed in the year",
        "Fighting on the Western Front ceased in the year"]),
    dict(id='y.ww2start', domain='history', value=1939, kind='year', stems=[
        "The Second World War began in the year",
        "Germany invaded Poland in the year",
        "Britain and France declared war on Germany in the year"]),
    dict(id='y.cuba', domain='history', value=1962, kind='year', stems=[
        "The Cuban Missile Crisis took place in the year",
        "Soviet missiles were discovered in Cuba in the year",
        "The thirteen-day standoff over Cuba occurred in the year"]),
    dict(id='y.ussr', domain='history', value=1991, kind='year', stems=[
        "The Soviet Union was dissolved in the year",
        "The USSR formally ceased to exist in the year",
        "The Soviet flag was lowered over the Kremlin in the year"]),
    dict(id='y.waterloo', domain='history', value=1815, kind='year', stems=[
        "The Battle of Waterloo was fought in the year",
        "Napoleon was finally defeated in Belgium in the year",
        "Wellington and Blucher beat Napoleon in the year"]),
    dict(id='y.penicillin', domain='science', value=1928, kind='year', stems=[
        "Alexander Fleming discovered penicillin in the year",
        "The mould that became penicillin was noticed in the year",
        "Penicillin was first observed at St Mary's in the year"]),
    dict(id='y.origin', domain='science', value=1859, kind='year', stems=[
        "Darwin published 'On the Origin of Species' in the year",
        "'On the Origin of Species' first appeared in the year",
        "Darwin's book on natural selection was published in the year"]),
    dict(id='y.heart', domain='science', value=1967, kind='year', stems=[
        "The first human heart transplant was performed in the year",
        "Christiaan Barnard transplanted a human heart in the year",
        "The first successful heart transplant took place in the year"]),
    dict(id='y.iphone', domain='science', value=2007, kind='year', stems=[
        "The first iPhone was released in the year",
        "Apple launched its first smartphone in the year",
        "The original iPhone went on sale in the year"]),

    # smaller magnitudes, so the contrast is not confined to four-digit years
    dict(id='n.leapyear', domain='counting', value=366, kind='count', stems=[
        "The number of days in a leap year is",
        "A leap year contains a total of days numbering",
        "The count of days in a leap year is"]),
    dict(id='n.circle', domain='counting', value=360, kind='count', stems=[
        "The number of degrees in a full circle is",
        "A complete rotation measures in degrees",
        "The degrees in one full turn number"]),
    dict(id='n.states', domain='counting', value=50, kind='count', stems=[
        "The number of states in the United States is",
        "The United States is made up of states numbering",
        "The count of US states is"]),
    dict(id='n.bones', domain='counting', value=206, kind='count', stems=[
        "The number of bones in the adult human body is",
        "An adult human skeleton contains bones numbering",
        "The count of bones in an adult human is"]),
    dict(id='n.piano', domain='counting', value=88, kind='count', stems=[
        "The number of keys on a standard piano is",
        "A full-size piano keyboard has keys numbering",
        "The count of keys on a standard piano is"]),
    dict(id='n.chess', domain='counting', value=64, kind='count', stems=[
        "The number of squares on a chessboard is",
        "A chessboard contains squares numbering",
        "The count of squares on a chessboard is"]),
    dict(id='n.chromosomes', domain='counting', value=46, kind='count', stems=[
        "The number of chromosomes in a human cell is",
        "A typical human cell contains chromosomes numbering",
        "The count of human chromosomes is"]),
    dict(id='n.iron', domain='science', value=26, kind='count', stems=[
        "The atomic number of iron is",
        "Iron's position by proton count in the periodic table is",
        "The number of protons in an iron nucleus is"]),
    dict(id='n.gold', domain='science', value=79, kind='count', stems=[
        "The atomic number of gold is",
        "Gold's position by proton count in the periodic table is",
        "The number of protons in a gold nucleus is"]),
    dict(id='n.boiling', domain='science', value=100, kind='count', stems=[
        "The boiling point of water in degrees Celsius is",
        "Water boils at sea level at a Celsius temperature of",
        "In Celsius, water's boiling point is"]),
]

# The computed class must have its operands somewhere, and a first attempt put
# them only in its own question. That made DIGIT COUNT a perfect cue: 2 numbers
# for `computed`, 1 for `retrieved`, and a digit-count classifier scored 1.000
# against routing's 0.996 — the condition was unreadable.
#
# So the operands live in a prefix that BOTH classes carry verbatim, and the
# computed question refers back to them instead of restating them. The context
# is then literally identical within a pair; only the question differs, and both
# prompts contain exactly the same two numbers.
#
#   shared     "Note the values 3 and 1645."
#   retrieved  "... The Treaty of Westphalia was signed in the year"  -> 1648
#   computed   "... Added together, the two values above come to"     -> 1648
#
# Phrasings are wordy on purpose: a terse "Their sum is" would make the computed
# question far shorter than any fact question and hand back a length cue in
# place of the digit cue.
SUMS = (
    "Taking those two values together, their sum is",
    "Added together, the two values above come to",
    "The total obtained by summing the two values is",
)

# offsets vary by paraphrase so the suite is not one repeated sum
OFFSETS = (3, 5, 7)


def build_computation():
    """`retrieved` vs `computed`, answer token identical within each pair."""
    out = []
    for _fi, f in enumerate(NUMERIC):
        v = f['value']
        for i, stem in enumerate(f['stems']):
            n = OFFSETS[i % len(OFFSETS)]
            shared = f"Note the values {n} and {v - n}."
            for cls, text in (('retrieved', f"{shared} {stem}"),
                              ('computed', f"{shared} {SUMS[i % len(SUMS)]}")):
                out.append({
                    'probe_id': f"{f['id']}.{cls[:4]}{i}", 'fact_id': f['id'],
                    'domain': f['domain'], 'cls': cls, 'para': i,
                    'dkind': f['kind'], 'matched': True, 'suite': 'computation',
                    'atype': 'num', 'stem': text, 'answer': ' ' + str(v)})
    return out


# ---------------------------------------------------------------------------
# R4 — rung 3: does an individual FACT have a routing address?
#
# The naive version of this test is confounded and would have looked like a
# success. Ask "which fact is this?" over paraphrases of "the capital of
# Australia" and every one of them contains the word *Australia*; a classifier
# reading topic would score well and prove nothing.
#
# A CROSSED GRID controls it from both sides. Each cell is one fact:
#
#                capital   currency   language   continent
#     France     Paris     euro       French     Europe
#     Japan      Tokyo     yen        Japanese   Asia
#
#   same entity, different relation  -> vocabulary nearly identical, fact
#                                       differs. Rules out topic.
#   same relation, different entity  -> template identical, fact differs.
#                                       Rules out question form.
#
# A routing pattern that resolves a cell under BOTH constraints is addressing
# the fact rather than its surroundings. Evaluation must hold out a whole
# PARAPHRASE (train on two wordings, test on a third), and must be compared
# against a bag-of-words baseline on the prompt — otherwise "routing identifies
# the fact" may just be "routing re-encodes the prompt".
GRID_RELATIONS = {
    'capital': ("The capital of {e} is",
                "{e}'s seat of government is the city of",
                "The city where {e}'s national government sits is"),
    'currency': ("The currency of {e} is the",
                 "Money in {e} is denominated in the",
                 "{e}'s national unit of currency is the"),
    'language': ("The main language spoken in {e} is",
                 "{e}'s official language is",
                 "The language most people in {e} speak is"),
    'continent': ("The continent containing {e} is",
                  "{e} is located on the continent of",
                  "Geographically, {e} lies within the continent of"),
}

GRID = [
    dict(e='France', capital='Paris', currency='euro', language='French',
         continent='Europe'),
    dict(e='Japan', capital='Tokyo', currency='yen', language='Japanese',
         continent='Asia'),
    dict(e='Brazil', capital='Brasilia', currency='real', language='Portuguese',
         continent='America'),
    dict(e='Egypt', capital='Cairo', currency='pound', language='Arabic',
         continent='Africa'),
    dict(e='India', capital='Delhi', currency='rupee', language='Hindi',
         continent='Asia'),
    dict(e='Poland', capital='Warsaw', currency='zloty', language='Polish',
         continent='Europe'),
    dict(e='Kenya', capital='Nairobi', currency='shilling', language='Swahili',
         continent='Africa'),
    dict(e='Norway', capital='Oslo', currency='krone', language='Norwegian',
         continent='Europe'),
    dict(e='Vietnam', capital='Hanoi', currency='dong', language='Vietnamese',
         continent='Asia'),
    dict(e='Peru', capital='Lima', currency='sol', language='Spanish',
         continent='America'),
    dict(e='Turkey', capital='Ankara', currency='lira', language='Turkish',
         continent='Asia'),
    dict(e='Sweden', capital='Stockholm', currency='krona', language='Swedish',
         continent='Europe'),
]


def build_grid():
    """One probe per (entity, relation, paraphrase). `fact_id` is the cell."""
    out = []
    for g in GRID:
        for rel, templates in GRID_RELATIONS.items():
            for i, t in enumerate(templates):
                out.append({
                    'probe_id': f"{g['e']}.{rel}.{i}",
                    'fact_id': f"{g['e']}.{rel}", 'entity': g['e'],
                    'relation': rel, 'domain': 'grid', 'cls': rel,
                    'para': i, 'dkind': None, 'matched': True,
                    'suite': 'grid', 'atype': 'word',
                    'stem': t.format(e=g['e']), 'answer': ' ' + g[rel]})
    return out


# ---------------------------------------------------------------------------
# R5 — does routing distinguish retrieval from FABRICATION?
#
# The existing captures cannot answer this: they were built from facts the
# model reliably knows, so the computation suite produced 2 failures out of 228.
# Testing whether routing predicts fabrication needs probes that ELICIT it.
#
# Three familiarity levels through IDENTICAL templates, because the confound
# here is obvious and fatal — if the fabricated probes were also worded
# differently, any separation would be wording. Only the entity changes:
#
#   known      well-attested; the model should retrieve
#   obscure    real, rarely attested; the model may or may not know
#   fictional  invented, plausible-sounding; there IS no fact, so any confident
#              answer is fabrication by construction
#
# Fictional names are built to look like the real ones (Latinate country-shaped
# forms) so the contrast is familiarity rather than orthography. That is a
# mitigation, not a guarantee — the analysis still has to check that the
# separation is not simply "this token sequence is rare", which is why the
# `obscure` middle class exists: it is real but also rare, so a pure rarity
# detector should group it with `fictional` while a fabrication detector should
# group it with `known` when the model answers correctly.
HALLUCINATION = [
    ('known', 'France', 'Paris', 'euro'),
    ('known', 'Japan', 'Tokyo', 'yen'),
    ('known', 'Egypt', 'Cairo', 'pound'),
    ('known', 'Norway', 'Oslo', 'krone'),
    ('known', 'Poland', 'Warsaw', 'zloty'),
    ('known', 'Brazil', 'Brasilia', 'real'),
    ('known', 'Turkey', 'Ankara', 'lira'),
    ('known', 'Sweden', 'Stockholm', 'krona'),
    ('obscure', 'Kiribati', 'Tarawa', 'dollar'),
    ('obscure', 'Bhutan', 'Thimphu', 'ngultrum'),
    ('obscure', 'Comoros', 'Moroni', 'franc'),
    ('obscure', 'Suriname', 'Paramaribo', 'dollar'),
    ('obscure', 'Eswatini', 'Mbabane', 'lilangeni'),
    ('obscure', 'Vanuatu', 'Vila', 'vatu'),
    ('obscure', 'Lesotho', 'Maseru', 'loti'),
    ('obscure', 'Tajikistan', 'Dushanbe', 'somoni'),
    ('fictional', 'Verdania', None, None),
    ('fictional', 'Kaltrovia', None, None),
    ('fictional', 'Mersonia', None, None),
    ('fictional', 'Tolvenia', None, None),
    ('fictional', 'Brashaland', None, None),
    ('fictional', 'Quenteria', None, None),
    ('fictional', 'Ardennica', None, None),
    ('fictional', 'Sallovia', None, None),
]


def build_hallucination():
    """Identical templates across familiarity levels; only the entity varies."""
    out = []
    for level, entity, cap, cur in HALLUCINATION:
        for rel, answer in (('capital', cap), ('currency', cur)):
            for i, t in enumerate(GRID_RELATIONS[rel][:2]):
                out.append({
                    'probe_id': f"{entity}.{rel}.{i}",
                    'fact_id': f"{entity}.{rel}", 'entity': entity,
                    'relation': rel, 'domain': level, 'cls': level,
                    'para': i, 'dkind': None, 'matched': True,
                    'suite': 'hallucination', 'atype': 'word',
                    'stem': t.format(e=entity),
                    # fictional entities have no answer; a placeholder keeps the
                    # capture uniform and `correct` is meaningless for them
                    'answer': ' ' + (answer if answer else 'unknown')})
    return out


def all_probes(suite='mechanism'):
    if suite == 'computation':
        return build_computation()
    if suite == 'grid':
        return build_grid()
    if suite == 'grid2':
        return build_grid2()
    if suite == 'hallucination':
        return build_hallucination()
    if suite == 'grounding':
        return build_grounding()
    if suite == 'all':
        return (build() + build_unmatched() + build_grounding()
                + build_computation())
    return build() + build_unmatched()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--dump')
    ap.add_argument('--show', type=int, default=0,
                    help='print N example fact groups in full')
    a = ap.parse_args()

    ps = all_probes()
    if a.stats or not (a.dump or a.show):
        from collections import Counter
        print(f"{len(F)} facts · {len(ps)} probes")
        print(f"  by class   {dict(Counter(p['cls'] for p in ps))}")
        print(f"  by domain  {dict(Counter(p['domain'] for p in ps))}")
        print(f"  matched    {sum(p['matched'] for p in ps)} / {len(ps)}")
        print(f"  facts with both classes: "
              f"{sum(1 for f in F if any(p['fact_id'] == f['id'] and p['cls'] == 'derive' for p in ps))}"
              f"/{len(F)}")
        paras = Counter(p['fact_id'] for p in ps if p['cls'] == 'recall')
        print(f"  recall paraphrases per fact: "
              f"{min(paras.values())}-{max(paras.values())}")
    if a.show:
        for f in F[:a.show]:
            print(f"\n--- {f['id']} ({f['domain']}) ---")
            for p in ps:
                if p['fact_id'] == f['id']:
                    print(f"  [{p['cls']:6s}] {p['stem']}"
                          f"  ->{p['answer']!r}")
    if a.dump:
        Path(a.dump).write_text(json.dumps(ps, indent=1))
        print(f"  → {a.dump}")


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# R8 — the expanded grid, for rung 4 (do fact addresses cluster semantically?)
#
# R5's grid could not test rung 4 for a reason more basic than sample size: all
# twelve entities were countries, so it contained exactly ONE semantic domain
# and there was nothing to cluster. Testing whether addresses group by meaning
# needs multiple entity TYPES.
#
# Four domains, deliberately chosen to be semantically distinct with minimal
# shared vocabulary, each keeping the crossed design that made R5 interpretable:
#
#   same entity, different relation  -> vocabulary near-identical, fact differs
#   same relation, different entity  -> template identical, fact differs
#
# and now additionally:
#
#   same relation TYPE, different domain -> tests whether clustering is about
#                                           meaning or about question form
#
# THE CHAT WRAPPER IS KEPT UNCHANGED. R5b established the carried-state claim
# precisely because the suffix is byte-identical across probes, which is what
# makes accuracy at those positions impossible to explain by surface form.
# Varying the format here would forfeit that. The naturalistic-phrasing axis is
# served by `annotate.py` on free text instead, where it belongs.
GRID2 = {
    'country': {
        'entities': [
            dict(e='France', capital='Paris', currency='euro',
                 language='French', continent='Europe'),
            dict(e='Japan', capital='Tokyo', currency='yen',
                 language='Japanese', continent='Asia'),
            dict(e='Brazil', capital='Brasilia', currency='real',
                 language='Portuguese', continent='America'),
            dict(e='Egypt', capital='Cairo', currency='pound',
                 language='Arabic', continent='Africa'),
            dict(e='India', capital='New Delhi', currency='rupee',
                 language='Hindi', continent='Asia'),
            dict(e='Poland', capital='Warsaw', currency='zloty',
                 language='Polish', continent='Europe'),
            dict(e='Kenya', capital='Nairobi', currency='shilling',
                 language='Swahili', continent='Africa'),
            dict(e='Norway', capital='Oslo', currency='krone',
                 language='Norwegian', continent='Europe'),
            dict(e='Vietnam', capital='Hanoi', currency='dong',
                 language='Vietnamese', continent='Asia'),
            dict(e='Peru', capital='Lima', currency='sol',
                 language='Spanish', continent='America'),
            dict(e='Chile', capital='Santiago', currency='peso',
                 language='Spanish', continent='America'),
            dict(e='Sweden', capital='Stockholm', currency='krona',
                 language='Swedish', continent='Europe'),
        ],
        'relations': {
            'capital': ("The capital of {e} is",
                        "{e}'s seat of government is the city of",
                        "The city where {e}'s national government sits is"),
            'currency': ("The currency of {e} is the",
                         "Money in {e} is denominated in the",
                         "{e}'s national unit of currency is the"),
            'language': ("The main language spoken in {e} is",
                         "{e}'s official language is",
                         "The language most people in {e} speak is"),
            'continent': ("The continent containing {e} is",
                          "{e} is located on the continent of",
                          "Geographically, {e} lies within the continent of"),
        }},
    'element': {
        'entities': [
            dict(e='iron', symbol='Fe', number='26', category='metal',
                 state='solid'),
            dict(e='gold', symbol='Au', number='79', category='metal',
                 state='solid'),
            dict(e='oxygen', symbol='O', number='8', category='nonmetal',
                 state='gas'),
            dict(e='helium', symbol='He', number='2', category='noble',
                 state='gas'),
            dict(e='carbon', symbol='C', number='6', category='nonmetal',
                 state='solid'),
            dict(e='sodium', symbol='Na', number='11', category='metal',
                 state='solid'),
            dict(e='mercury', symbol='Hg', number='80', category='metal',
                 state='liquid'),
            dict(e='nitrogen', symbol='N', number='7', category='nonmetal',
                 state='gas'),
            dict(e='copper', symbol='Cu', number='29', category='metal',
                 state='solid'),
            dict(e='sulfur', symbol='S', number='16', category='nonmetal',
                 state='solid'),
            dict(e='neon', symbol='Ne', number='10', category='noble',
                 state='gas'),
            dict(e='zinc', symbol='Zn', number='30', category='metal',
                 state='solid'),
        ],
        'relations': {
            'symbol': ("The chemical symbol for {e} is",
                       "On the periodic table, {e} is written as",
                       "Chemists abbreviate {e} as"),
            'number': ("The atomic number of {e} is",
                       "The number of protons in a {e} nucleus is",
                       "{e} sits in the periodic table at atomic number"),
            'category': ("As an element, {e} is classified as a",
                         "In elemental terms {e} counts as a",
                         "Chemically, {e} belongs to the group known as"),
            'state': ("At room temperature, {e} is a",
                      "Under ordinary conditions {e} exists as a",
                      "The state of {e} at room temperature is"),
        }},
    'author': {
        'entities': [
            dict(e='Orwell', nationality='British', century='twentieth',
                 language='English', genre='dystopia'),
            dict(e='Austen', nationality='British', century='nineteenth',
                 language='English', genre='romance'),
            dict(e='Tolstoy', nationality='Russian', century='nineteenth',
                 language='Russian', genre='epic'),
            dict(e='Cervantes', nationality='Spanish', century='seventeenth',
                 language='Spanish', genre='satire'),
            dict(e='Goethe', nationality='German', century='eighteenth',
                 language='German', genre='drama'),
            dict(e='Borges', nationality='Argentine', century='twentieth',
                 language='Spanish', genre='fantasy'),
            dict(e='Flaubert', nationality='French', century='nineteenth',
                 language='French', genre='realism'),
            dict(e='Kafka', nationality='Czech', century='twentieth',
                 language='German', genre='absurdism'),
            dict(e='Dante', nationality='Italian', century='fourteenth',
                 language='Italian', genre='poetry'),
            dict(e='Dostoevsky', nationality='Russian', century='nineteenth',
                 language='Russian', genre='psychological'),
            dict(e='Ibsen', nationality='Norwegian', century='nineteenth',
                 language='Norwegian', genre='drama'),
            dict(e='Chekhov', nationality='Russian', century='nineteenth',
                 language='Russian', genre='drama'),
        ],
        'relations': {
            'nationality': ("The writer {e} was by nationality",
                            "{e} held the nationality",
                            "As an author, {e} is described as"),
            'century': ("The writer {e} worked chiefly in the",
                        "{e}'s literary career fell in the",
                        "The century in which {e} wrote was the"),
            'language': ("The writer {e} composed chiefly in",
                         "{e} wrote in the language",
                         "The language of {e}'s works is"),
            'genre': ("The writer {e} is chiefly associated with the genre of",
                      "{e}'s work is usually classed as",
                      "In genre terms, {e} is known for"),
        }},
    'composer': {
        'entities': [
            dict(e='Bach', nationality='German', era='baroque', form='fugue',
                 century='eighteenth'),
            dict(e='Mozart', nationality='Austrian', era='classical',
                 form='symphony', century='eighteenth'),
            dict(e='Beethoven', nationality='German', era='classical',
                 form='symphony', century='nineteenth'),
            dict(e='Chopin', nationality='Polish', era='romantic',
                 form='nocturne', century='nineteenth'),
            dict(e='Verdi', nationality='Italian', era='romantic', form='opera',
                 century='nineteenth'),
            dict(e='Debussy', nationality='French', era='impressionist',
                 form='prelude', century='twentieth'),
            dict(e='Stravinsky', nationality='Russian', era='modernist',
                 form='ballet', century='twentieth'),
            dict(e='Vivaldi', nationality='Italian', era='baroque',
                 form='concerto', century='eighteenth'),
            dict(e='Handel', nationality='German', era='baroque',
                 form='oratorio', century='eighteenth'),
            dict(e='Wagner', nationality='German', era='romantic', form='opera',
                 century='nineteenth'),
            dict(e='Liszt', nationality='Hungarian', era='romantic',
                 form='rhapsody', century='nineteenth'),
            dict(e='Sibelius', nationality='Finnish', era='romantic',
                 form='symphony', century='twentieth'),
        ],
        'relations': {
            'nationality': ("The composer {e} was by nationality",
                            "{e} held the nationality",
                            "As a composer, {e} is described as"),
            'era': ("The composer {e} belongs to the musical era known as",
                    "{e}'s music is classed as",
                    "In period terms, {e} is a composer of the"),
            'form': ("The musical form most associated with {e} is the",
                     "{e} is best known for writing the",
                     "The genre {e} is chiefly remembered for is the"),
            'century': ("The composer {e} worked chiefly in the",
                        "{e}'s career fell in the",
                        "The century in which {e} composed was the"),
        }},
}


def build_grid2():
    """One probe per (domain, entity, relation, paraphrase)."""
    out = []
    for dom, spec in GRID2.items():
        for g in spec['entities']:
            for rel, templates in spec['relations'].items():
                for i, t in enumerate(templates):
                    out.append({
                        'probe_id': f"{dom}.{g['e']}.{rel}.{i}",
                        'fact_id': f"{dom}.{g['e']}.{rel}",
                        'entity': g['e'], 'relation': rel, 'domain': dom,
                        'cls': rel, 'para': i, 'dkind': None, 'matched': True,
                        'suite': 'grid2', 'atype': 'word',
                        'stem': t.format(e=g['e']), 'answer': ' ' + g[rel]})
    return out
