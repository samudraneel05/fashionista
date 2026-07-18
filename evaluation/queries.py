"""100 test queries for evaluation — 5 categories × 20 variations each.

Base queries from the assignment:
1. Attribute Specific: "A person in a bright yellow raincoat."
2. Contextual/Place: "Professional business attire inside a modern office."
3. Complex Semantic: "Someone wearing a blue shirt sitting on a park bench."
4. Style Inference: "Casual weekend outfit for a city walk."
5. Compositional: "A red tie and a white shirt in a formal setting."
"""

BASE_QUERIES = {
    "attribute": "A person in a bright yellow raincoat.",
    "contextual": "Professional business attire inside a modern office.",
    "complex": "Someone wearing a blue shirt sitting on a park bench.",
    "style": "Casual weekend outfit for a city walk.",
    "compositional": "A red tie and a white shirt in a formal setting.",
}

# 20 variations per category (paraphrases + attribute swaps)
QUERY_VARIATIONS = {
    "attribute": [
        "A person in a bright yellow raincoat.",
        "Someone wearing a yellow rain jacket.",
        "A person dressed in a bright yellow coat.",
        "Someone in a yellow waterproof jacket.",
        "A person wearing a bright yellow raincoat outdoors.",
        "A person in a vivid yellow raincoat.",
        "Someone wearing a yellow rain slicker.",
        "A person dressed in a yellow rain poncho.",
        "Someone in a bright yellow outerwear jacket.",
        "A person wearing a yellow raincoat in the rain.",
        "A person in a bright yellow windbreaker.",
        "Someone wearing a yellow trench coat.",
        "A person dressed in a bright yellow anorak.",
        "Someone in a yellow rain shell jacket.",
        "A person wearing a bright yellow parka.",
        "A person in a yellow rain jacket with hood.",
        "Someone wearing a bright yellow waterproof coat.",
        "A person dressed in a yellow rain mac.",
        "Someone in a bright yellow cagoule.",
        "A person wearing a yellow rain overcoat.",
    ],
    "contextual": [
        "Professional business attire inside a modern office.",
        "A person in formal work clothes in an office building.",
        "Someone wearing business professional clothing at work.",
        "A person in office attire in a corporate setting.",
        "Professional dress code in a modern workplace.",
        "Someone in a suit in an office environment.",
        "A person wearing business formal in an office.",
        "Corporate attire in a modern office space.",
        "A person in professional clothing at their desk.",
        "Someone wearing formal business wear in an office.",
        "A person dressed for a business meeting in an office.",
        "Professional office wear in a modern building.",
        "Someone in business attire in a corporate office.",
        "A person wearing a suit and tie in an office setting.",
        "Formal business clothing in a workplace.",
        "A person in work-appropriate attire in an office.",
        "Someone dressed professionally in a modern office.",
        "A person in corporate wear in an office building.",
        "Business formal attire in a contemporary office.",
        "A person wearing office-appropriate formal clothing.",
    ],
    "complex": [
        "Someone wearing a blue shirt sitting on a park bench.",
        "A person in a blue shirt on a bench in the park.",
        "Someone in a blue top sitting on a park bench outdoors.",
        "A person wearing blue and sitting on a bench in nature.",
        "Someone with a blue shirt resting on a park bench.",
        "A person in a blue shirt seated on a bench in a park.",
        "Someone wearing a blue shirt on a bench surrounded by trees.",
        "A person in blue sitting on a bench in a green park.",
        "Someone in a blue shirt on a park bench outside.",
        "A person wearing a blue top on a bench in a garden.",
        "Someone in a blue shirt sitting outdoors on a bench.",
        "A person in blue clothing on a park bench.",
        "Someone wearing blue and sitting on a bench in nature.",
        "A person in a blue shirt on a wooden park bench.",
        "Someone in a blue top seated on a bench in a park setting.",
        "A person wearing a blue shirt relaxing on a park bench.",
        "Someone in blue on a bench in an outdoor park.",
        "A person with a blue shirt on a bench in a green space.",
        "Someone wearing a blue shirt on a bench in a public park.",
        "A person in a blue shirt sitting on a bench outdoors.",
    ],
    "style": [
        "Casual weekend outfit for a city walk.",
        "A person in relaxed weekend wear walking in the city.",
        "Someone in casual clothes for a stroll around town.",
        "A person wearing a laid-back outfit for a city stroll.",
        "Casual everyday attire for walking around the city.",
        "Someone in comfortable weekend clothes in an urban setting.",
        "A person in relaxed casual wear for a city outing.",
        "Weekend casual style for a walk downtown.",
        "A person in laid-back weekend fashion on city streets.",
        "Someone wearing casual clothes for a city walk.",
        "A person in comfy weekend attire walking in town.",
        "Casual street style for a weekend city stroll.",
        "Someone in relaxed everyday wear for walking in the city.",
        "A person dressed casually for a weekend walk around town.",
        "Informal weekend outfit for an urban stroll.",
        "Someone in casual weekend clothing walking in the city.",
        "A person in relaxed attire for a city walk on weekends.",
        "Casual comfortable outfit for walking around town.",
        "Someone in laid-back clothes for a weekend city walk.",
        "A person wearing casual weekend style for a stroll in the city.",
    ],
    "compositional": [
        "A red tie and a white shirt in a formal setting.",
        "Someone wearing a white shirt with a red tie in a formal environment.",
        "A person in a white shirt and red tie at a formal event.",
        "Someone with a red tie over a white shirt in formal attire.",
        "A person wearing a white button-down with a red tie formally.",
        "A red tie paired with a white shirt in a formal context.",
        "Someone in a white shirt and red tie in a formal setting.",
        "A person with a white shirt and red tie in formal wear.",
        "Someone wearing a red tie with a white shirt at a formal occasion.",
        "A white shirt and red tie in a formal business setting.",
        "A person in formal attire with a red tie and white shirt.",
        "Someone wearing a white shirt with red tie in a formal environment.",
        "A red tie and white shirt combination in a formal setting.",
        "A person dressed formally with a white shirt and red tie.",
        "Someone in a white shirt wearing a red tie in a formal setting.",
        "A red tie on a white shirt in a formal context.",
        "A person with a red tie and white shirt in formal clothing.",
        "Someone wearing a white shirt and a red tie to a formal event.",
        "A white shirt paired with a red tie in formal attire.",
        "A person in a formal setting wearing a white shirt and red tie.",
    ],
}


def get_all_queries() -> list:
    """Get all 100 test queries with category labels."""
    queries = []
    for category, variations in QUERY_VARIATIONS.items():
        for i, query in enumerate(variations):
            queries.append({
                "id": f"{category}_{i:02d}",
                "category": category,
                "query": query,
                "is_base": i == 0,
            })
    return queries


def get_base_queries() -> list:
    """Get the 5 base evaluation queries."""
    queries = []
    for category, query in BASE_QUERIES.items():
        queries.append({
            "id": f"{category}_00",
            "category": category,
            "query": query,
            "is_base": True,
        })
    return queries
