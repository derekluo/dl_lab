import os
import random
import json

def generate_test_texts():
    """Generate diverse test texts for each category"""

    # Comprehensive vocabulary for each category
    categories = {
        "Technology": {
            "words": [
                "computer", "software", "programming", "algorithm", "database", "artificial", "intelligence",
                "machine", "learning", "neural", "network", "code", "development", "system", "technology",
                "application", "framework", "library", "function", "variable", "debugging", "testing",
                "deployment", "server", "client", "API", "interface", "protocol", "encryption", "security",
                "cybersecurity", "blockchain", "cryptocurrency", "cloud", "computing", "virtualization",
                "containers", "microservices", "DevOps", "agile", "scrum", "version", "control", "git"
            ],
            "templates": [
                "The {adj} {noun} uses advanced {tech} for {purpose}",
                "Developers are implementing {tech} to improve {noun} performance",
                "This {adj} software framework provides {tech} capabilities",
                "The {noun} system integrates with {tech} for better {purpose}",
                "Modern {tech} enables efficient {noun} processing and {purpose}"
            ]
        },
        "Sports": {
            "words": [
                "football", "basketball", "soccer", "baseball", "tennis", "volleyball", "hockey", "swimming",
                "running", "marathon", "championship", "tournament", "league", "team", "player", "athlete",
                "coach", "training", "practice", "game", "match", "score", "goal", "touchdown", "slam", "dunk",
                "home", "run", "ace", "serve", "penalty", "foul", "referee", "stadium", "arena", "field",
                "court", "track", "pool", "gym", "fitness", "exercise", "workout", "competition", "victory"
            ],
            "templates": [
                "The {adj} {team} won the {competition} with a final score of {score}",
                "The {player} made an incredible {move} during the {game}",
                "This season's {competition} features the best {teams} in the league",
                "The {adj} {athlete} broke the record in {sport} competition",
                "Fans gathered at the {venue} to watch the championship {game}"
            ]
        },
        "Science": {
            "words": [
                "research", "experiment", "hypothesis", "theory", "analysis", "study", "discovery", "method",
                "laboratory", "scientist", "professor", "physics", "chemistry", "biology", "mathematics",
                "statistics", "data", "observation", "measurement", "microscope", "telescope", "molecule",
                "atom", "cell", "gene", "protein", "enzyme", "reaction", "catalyst", "compound", "element",
                "periodic", "table", "quantum", "mechanics", "relativity", "evolution", "ecosystem", "species"
            ],
            "templates": [
                "Scientists conducted {adj} experiments to test their hypothesis about {subject}",
                "The research team made a groundbreaking discovery in {field} science",
                "This {adj} study reveals important insights about {phenomenon}",
                "Laboratory analysis shows that {subject} behaves differently under {conditions}",
                "The {adj} method provides accurate measurements of {property}"
            ]
        },
        "Music": {
            "words": [
                "music", "song", "album", "artist", "musician", "singer", "composer", "producer", "band",
                "orchestra", "symphony", "concert", "performance", "stage", "audience", "melody", "harmony",
                "rhythm", "beat", "tempo", "chord", "note", "scale", "instrument", "guitar", "piano", "violin",
                "drums", "saxophone", "trumpet", "flute", "recording", "studio", "microphone", "sound",
                "audio", "genre", "classical", "jazz", "rock", "pop", "country", "hip-hop", "electronic"
            ],
            "templates": [
                "The {adj} musician performed a beautiful {piece} on the {instrument}",
                "This {genre} album features {adj} melodies and {rhythm} rhythms",
                "The {adj} concert showcased talented artists and their {instruments}",
                "The composer created a {adj} symphony that captures the essence of {emotion}",
                "During the {adj} performance, the audience enjoyed the {harmony} between instruments"
            ]
        },
        "Food": {
            "words": [
                "food", "cooking", "recipe", "ingredient", "meal", "dish", "restaurant", "chef", "kitchen",
                "cuisine", "flavor", "taste", "delicious", "spicy", "sweet", "salty", "bitter", "umami",
                "fresh", "organic", "healthy", "nutritious", "protein", "vegetable", "fruit", "meat", "fish",
                "chicken", "beef", "pasta", "rice", "bread", "cheese", "milk", "egg", "butter", "olive", "oil",
                "herbs", "spices", "garlic", "onion", "tomato", "potato", "carrot", "broccoli", "spinach"
            ],
            "templates": [
                "The chef prepared a {adj} meal with fresh {ingredients} from the market",
                "This {adj} recipe combines {ingredient1} and {ingredient2} for a delicious {dish}",
                "The restaurant serves {adj} cuisine featuring local {ingredients}",
                "The {adj} dish has a perfect balance of {flavor1} and {flavor2} flavors",
                "Home cooks love this {adj} recipe because it uses simple {ingredients}"
            ]
        }
    }

    # Additional words for templates
    adjectives = ["amazing", "excellent", "outstanding", "innovative", "creative", "powerful", "efficient",
                 "advanced", "sophisticated", "modern", "traditional", "popular", "successful", "impressive"]

    # Generate test texts
    test_texts = {}

    for category, data in categories.items():
        texts = []
        words = data["words"]
        templates = data["templates"]

        # Generate 10 texts per category using different approaches
        for i in range(10):
            if i < 5:  # Use templates
                template = random.choice(templates)
                # Fill template with appropriate words
                text = template.format(
                    adj=random.choice(adjectives),
                    noun=random.choice(words[:20]),  # Use first 20 words as nouns
                    tech=random.choice(words[10:30]),
                    purpose="analysis and optimization",
                    team=random.choice(["team", "group", "squad"]),
                    competition=random.choice(["championship", "tournament", "league"]),
                    score=f"{random.randint(10,99)}-{random.randint(10,99)}",
                    player=random.choice(["player", "athlete", "star"]),
                    move=random.choice(["shot", "play", "move"]),
                    game=random.choice(["game", "match", "contest"]),
                    teams=random.choice(["teams", "clubs", "squads"]),
                    athlete=random.choice(["athlete", "player", "competitor"]),
                    sport=random.choice(words[:10]),
                    venue=random.choice(["stadium", "arena", "field"]),
                    subject=random.choice(words[5:15]),
                    field=random.choice(["biological", "chemical", "physical"]),
                    phenomenon=random.choice(["cellular behavior", "molecular structure", "quantum effects"]),
                    conditions=random.choice(["extreme temperatures", "high pressure", "controlled environment"]),
                    property=random.choice(["temperature", "pressure", "density"]),
                    piece=random.choice(["melody", "symphony", "concerto"]),
                    instrument=random.choice(words[20:30]),
                    genre=random.choice(["jazz", "classical", "rock"]),
                    rhythm=random.choice(["complex", "simple", "syncopated"]),
                    instruments=random.choice(["guitars", "violins", "pianos"]),
                    emotion=random.choice(["joy", "melancholy", "triumph"]),
                    harmony=random.choice(["beautiful harmony", "perfect blend", "amazing synergy"]),
                    ingredients=random.choice(["ingredients", "produce", "items"]),
                    dish=random.choice(["dish", "meal", "creation"]),
                    ingredient1=random.choice(words[15:25]),
                    ingredient2=random.choice(words[25:35]),
                    flavor1=random.choice(["sweet", "spicy", "savory"]),
                    flavor2=random.choice(["tangy", "rich", "fresh"])
                )
            else:  # Generate free-form sentences
                sentence_words = random.sample(words, random.randint(8, 15))
                text = " ".join(sentence_words)
                # Make it more sentence-like
                text = text.capitalize() + " and " + " ".join(random.sample(words, random.randint(3, 8)))

            texts.append(text)

        test_texts[category] = texts

    return test_texts

def save_test_texts(test_texts):
    """Save test texts to files"""
    # Create test_texts directory if it doesn't exist
    if not os.path.exists('test_texts'):
        os.makedirs('test_texts')

    # Save individual category files
    for category, texts in test_texts.items():
        filename = f'test_texts/{category.lower()}_texts.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"{i+1}. {text}\n")
        print(f'Generated {filename} with {len(texts)} texts')

    # Save combined JSON file for easy loading
    with open('test_texts/all_test_texts.json', 'w', encoding='utf-8') as f:
        json.dump(test_texts, f, indent=2, ensure_ascii=False)

    print('Generated test_texts/all_test_texts.json with all categories')

def generate_mixed_test_set():
    """Generate a mixed test set with labels for evaluation"""
    test_texts = generate_test_texts()

    mixed_data = []
    category_to_id = {
        "Technology": 0,
        "Sports": 1,
        "Science": 2,
        "Music": 3,
        "Food": 4
    }

    for category, texts in test_texts.items():
        label = category_to_id[category]
        for text in texts:
            mixed_data.append({
                "text": text,
                "label": label,
                "category": category
            })

    # Shuffle the data
    random.shuffle(mixed_data)

    # Save mixed test set
    with open('test_texts/mixed_test_set.json', 'w', encoding='utf-8') as f:
        json.dump(mixed_data, f, indent=2, ensure_ascii=False)

    print(f'Generated test_texts/mixed_test_set.json with {len(mixed_data)} labeled examples')

    return mixed_data

if __name__ == '__main__':
    print("Generating test texts for transformer text classifier...")

    # Set random seed for reproducible results
    random.seed(42)

    # Generate and save test texts
    test_texts = generate_test_texts()
    save_test_texts(test_texts)

    # Generate mixed test set
    mixed_data = generate_mixed_test_set()

    print(f"\nGenerated test data:")
    for category in test_texts.keys():
        print(f"  {category}: {len(test_texts[category])} texts")

    print(f"\nTotal: {len(mixed_data)} test examples across 5 categories")
    print("\nFiles created:")
    print("  - test_texts/[category]_texts.txt (individual category files)")
    print("  - test_texts/all_test_texts.json (all categories)")
    print("  - test_texts/mixed_test_set.json (shuffled labeled data)")
