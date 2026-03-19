"""
Domain constants for German Timber Market Intelligence.

This file is a vocabulary reference only.
It does NOT score, threshold, or gate sources.
The LLM uses its own judgment to decide relevance.
These lists exist only to help tools build good search queries
and to give the LLM context about what the domain covers.
"""

# Core timber / wood terms (EN + DE)
TIMBER_KEYWORDS: list[str] = [
    # Wood products
    "timber",
    "lumber",
    "wood",
    "holz",
    "bauholz",
    "schnittholz",
    "rundholz",
    "rohholz",
    "softwood",
    "hardwood",
    "sawnwood",
    "plywood",
    "sperrholz",
    "pellet",
    "biomasse",
    # Engineered wood
    "clt",
    "cross-laminated",
    "brettsperrholz",
    "mass timber",
    "glulam",
    "brettschichtholz",
    "leimholz",
    "ingenieurholz",
    "holzwerkstoff",
    "osb",
    "mdf",
    "spanplatte",
    # Industry
    "sawmill",
    "sägewerk",
    "saegewerk",
    "forestry",
    "forstwirtschaft",
    "logging",
    "holzeinschlag",
    "holzwirtschaft",
    "holzindustrie",
    "holzverarbeitung",
    "holzverarbeiter",
    # Market
    "holzpreis",
    "holzpreise",
    "holzmarkt",
    "timber price",
    "lumber price",
    "wood price",
    "holzhandel",
    "holznachfrage",
    "holzversorgung",
    "holzproduktion",
    # Trade
    "holzexport",
    "holzimport",
    "timber trade",
    "lumber tariff",
    "log export",
    "zölle",
    "zoll",
    # Construction link
    "holzbau",
    "timber construction",
    "timber frame",
    "wohnungsbau",
    "baugenehmigung",
    "housing starts",
    "baukonjunktur",
    "baubranche",
    # Environmental / forestry
    "borkenkäfer",
    "borkenkaefer",
    "bark beetle",
    "waldschäden",
    "waldschaeden",
    "forest damage",
    "aufforstung",
    "reforestation",
    "sturmholz",
    "schadholz",
    "salvage logging",
    "wildfire",
    "waldbrand",
    "bundeswaldinventur",
    "waldumbau",
    # Policy
    "eudr",
    "entwaldungsverordnung",
    "timber regulation",
    "lieferkettengesetz",
    "bmel",
    "lieferkette",
]

#  Geographic scope keywords
GERMAN_SCOPE_KEYWORDS: list[str] = [
    "germany",
    "german",
    "deutschland",
    "deutsch",
    "bundesrepublik",
    "bayern",
    "bavaria",
    "baden-württemberg",
    "niedersachsen",
    "nordrhein-westfalen",
    "hessen",
    "sachsen",
    "thüringen",
    "brandenburg",
    "mecklenburg",
    "rheinland-pfalz",
    "schleswig-holstein",
    "saarland",
    "sachsen-anhalt",
    "bremen",
    "hamburg",
    "berlin",
    "eu timber",
    "eu forestry",
    "european timber",
    "european forestry",
    "eu holz",
    "european union",
    "österreich",
    "austria",
    "schweiz",
    "switzerland",
    # Implicit Germany-scope industry terms
    "holzmarkt",
    "holzbranche",
    "holzwirtschaft",
    "forstwirtschaft",
    "bundesregierung",
    "bundesministerium",
    "baubranche",
    "baugewerbe",
]

# Pure social patterns
SOCIAL_PATTERNS: list[str] = [
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "how are you",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
    "nice to meet",
    "see you",
]

#  MediaStack search keywords
MEDIASTACK_GERMAN_KEYWORDS: list[str] = [
    # Core market terms
    "Holzpreis",
    "Holzpreise",
    "Rundholz",
    "Bauholz",
    "Schnittholz",
    "Sägewerk",
    "Holzmarkt",
    "Rohholz",
    # Supply side
    "Borkenkäfer",
    "Schadholz",
    "Waldschäden",
    "Forstwirtschaft",
    "Sturmholz",
    # Demand side
    "Baugenehmigungen",
    "Wohnungsbau",
    "Baukonjunktur",
    "Holzbau",
    # Policy
    "EUDR",
    "Entwaldungsverordnung",
    "Lieferkettengesetz",
    "Biomasse",
    # Trade
    "Holzexport",
    "Holzimport",
]

MEDIASTACK_ENGLISH_KEYWORDS: list[str] = [
    "Lumber prices",
    "Timber prices",
    "Softwood lumber",
    "Sawmill",
    "Housing starts",
    "Building permits",
    "EUDR",
    "Mass timber",
    "CLT",
    "Bark beetle",
    "Timber trade",
    "Wood products",
]

# Trusted specialist domains for Tavily
TIMBER_SPECIALIST_DOMAINS: list[str] = [
    "timber-online.net",
    "holzkurier.com",
    "euwid-holz.de",
    "euwid.de",
    "holz-zentralblatt.com",
    "saegeindustrie.de",
    "dhwr.de",
    "gdholz.de",
    "afz-derwald.de",
    "holzbau-deutschland.de",
    "bmel.de",
    "fordaq.com",
    "forest-industries.com",
    "timber-trades-journal.com",
    "risiinfo.com",
]


# Simple helper used only for routing decisions


def is_timber_related(text: str) -> bool:
    """Lightweight check used only by the router to decide if query is in domain."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in TIMBER_KEYWORDS)


def is_germany_related(text: str) -> bool:
    """Lightweight check used only by router/analyzer."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in GERMAN_SCOPE_KEYWORDS)
