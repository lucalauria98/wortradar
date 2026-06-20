"""Pseudowoerter fuer den Wortschatztest (Ratekorrektur).

Plausibel klingende, aber nicht existierende englische Woerter. Wer hier
"kenne ich" klickt, raet - daraus wird die Guessing-Rate g geschaetzt,
die den logistischen Fit korrigiert (Ansatz wie bei LexTALE ueblich).

Jedes Wort wurde gegen wordfreq (Zipf == 0) und lemminflect
(keine bekannte Flexionsform) geprueft.
"""

PSEUDOWORDS = [
    "platery", "mensible", "kilpfound", "alberation", "plaudate",
    "crumperly", "fellickson", "dentling", "gorpleness", "brastion",
    "tilfered", "mowselish", "skornet", "delpherate", "quandlement",
    "frimsy", "norpitude", "clandical", "vetrapose", "lonshering",
    "spaviney", "trobbisher", "wernicle", "hastorial",
]
