#!/usr/bin/env python3
import itertools
import csv
from collections import defaultdict, Counter
from typing import List, Set, Dict, FrozenSet, Tuple, Optional, NamedTuple, Union
import networkx as nx
import config
from pydantic import BaseModel, Field
import math
import argparse
import yaml
import sys
import spacy
from tqdm import tqdm
import json

class Categorizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def get_mss(self, word: str, examples: List[str]) -> float:
        sse = 0.0
        lexeme = self.nlp.vocab[word]
        if not lexeme.has_vector:
            return 0.0
        count = 0
        for word in examples:
            other = self.nlp.vocab[word]
            if not other.has_vector:
                continue
            sse += lexeme.similarity(self.nlp.vocab[word]) ** 2
            count += 1
        return sse / count if count > 0 else 0.0

class Category(BaseModel):
    name: str = Field(..., description="Name of the category")
    examples: List[str] = Field(default_factory=list, description="Example words in the category")
    counterexamples: List[str] = Field(default_factory=list, description="Example words NOT in the category")
    threshold: float = Field(
        default=0.25,
        description="Threshold for categorization; words with a similarity score above this will be categorized"
    )
    freq_multiplier: float = Field(default=1.0, description="Frequency multiplier for words in this category")
    freq_offset: float = Field(default=0.0, description="Frequency offset for words in this category")

default_categories = [
    Category(
        name="programming",
        examples=[
            "computer",
            "data",
            "digital",
            "hardware",
            "internet",
            "network",
            "packet",
            "password",
            "server",
            "software",
        ],
        counterexamples=["mac", "dos", "suse", "kde", "vlc",],
        threshold=0.20,
        freq_multiplier=2.0,
        freq_offset=0.0
    ),
    Category(
        name="commands",
        examples=[
            "cd",
            "git",
            "grep",
            "ssh",
            "ls",
            "cp",
            "mv",
            "rm",
            "mkdir",
        ],
        counterexamples=["debian", "firefox", "gnome","mozilla",],
        threshold=0.10,
        freq_multiplier=2.0,
        freq_offset=0.0
    ),
    Category(
        name="graph_theory",
        examples=[
            "node",
            "edge",
            "graph",
            "digraph",
        ],
        counterexamples=[],
        threshold=0.25,
        freq_multiplier=3.0,
        freq_offset=0.0
    ),
    Category(
        name="names",
        examples=[
            "robert", "john", "kennedy", "ashley", "rebecca", "michael", "jones"
        ],
        counterexamples=[
            "smith", "fletcher", "brown", "jan", "tuesday", "friday",
            "june", "jun", "january", "mason", "genesis", "london", "paris"
        ],
        threshold=0.25,
        freq_multiplier=0.2,
        freq_offset=-0.001
    ),
]

class HeuristicValues(BaseModel):
    pairwise_cost: float = Field(default=3.0, description="Base cost for pressing more keys in a chord")
    different_direction_cost: float = Field(default=5.0, description="Cost for pressing keys in different directions with a single hand")
    missing_first_letter_cost: float = Field(default=10.0, description="Cost for not including the first letter of the word in the chord")
    missing_second_letter_cost: float = Field(default=5.0, description="Cost for not including the second letter of the word in the chord")
    missing_last_letter_cost: float = Field(default=2.0, description="Cost for not including the last letter of the word in the chord")
    missing_any_letter_cost: float = Field(default=2.0, description="Cost for not including any other letter of the word in the chord")
    extra_chord_cost: float = Field(default=200.0, description="Cost for including non-letter characters in a chord other than DUP")
    bad_letter_cost: float = Field(default=100.0, description="Cost for including a letter in a chord that is not in the word")
    dup_without_duplicates_cost: float = Field(default=200.0, description="Cost for including 'DUP' in a chord when there are no duplicate letters in the word")
    dup_without_chord_duplicates_cost: float = Field(default=30.0, description="Cost for including 'DUP' in a chord when none of the letters in the chord are duplicates in the word")
    base_weight: float = Field(default=0.2, description="Base weight for chord assignments, used to encourage all words to have chords, even if they are not optimal")
    modifier_relative_frequency: float = Field(default=30.0, description="How much more common a 'modified' word must be before we assign it its own chord instead of adding it to the lemma")
    frequency_exponent: float = Field(default=2.0, description="Exponent for frequency in the chord cost calculation, higher means more emphasis on frequency")

class Settings(BaseModel):
    top_n: int = Field(default=4000, description="Number of top words to consider")
    max_chord_size: int = Field(default=6, description="Maximum size of a chord")
    min_chord_size: int = Field(default=2, description="Minimum size of a chord")
    skip_single_letter_words: bool = Field(default=True, description="Whether to skip single-letter words when assigning chords")
    two_word_freq_threshold: float = Field(
        default=0.01,
        description="Threshold for two-word phrases; words with frequency below this will not be considered"
    )
    extra_chord_keys: List[str] = Field(
        default_factory=lambda: [
            "DUP",
            "`",
        ],
        description="Extra keys to consider for chords, e.g., 'DUP' for duplicates"
    )
    banned_chords: List[Union[List[str], str]] = Field(
        default_factory=lambda: [
            "sh",  # 'sh' is a common digraph, so we ban it
            "th",  # 'th' is a common digraph, so we ban it
            ["i", "DUP"],  # Conflicts with impulse chords
        ],
        description="Chords that should not be assigned to any word"
    )
    explicit_chords: Dict[str, Union[List[str],str]] = Field(
        default_factory=lambda: {
            "at": "ae",
            "been": "ben",
            # Add more explicit chords as needed
        },
        description="Explicit chords for specific words"
    )
    categories: List[Category] = Field(
        default_factory=lambda: default_categories,
        description="List of categories with their examples and thresholds"
    )
    heuristic: HeuristicValues = Field(
        default_factory=HeuristicValues,
        description="Heuristic values for chord cost calculations"
    )

    def model_post_init(self, __context):
        """
        Post-initialization hook to ensure that extra_chord_keys are unique and sorted,
        and that banned_chords and explicit_chords are converted to frozensets for immutability.
        """
        frozen_banned_chords = [frozenset(chord) for chord in self.banned_chords]
        frozen_explicit_chords = {word: frozenset(chord) for word, chord in self.explicit_chords.items()}
        frozen_unavailable_chords = set(frozen_banned_chords).union(set(frozen_explicit_chords.values()))
        object.__setattr__(self, 'frozen_banned_chords', frozen_banned_chords)
        object.__setattr__(self, 'frozen_explicit_chords', frozen_explicit_chords)
        object.__setattr__(self, 'frozen_unavailable_chords', frozen_unavailable_chords)

def load_word_list(settings: Settings, path: str) -> Dict[str, float]:
    """Load a frequency-sorted word list from a CSV file with word,count format."""
    word_freqs = {}
    max_freq = 0
    delimiter = '\t' if path.endswith('.tsv') else ','
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) == 2:
                    try:
                        word, count = row[0], float(row[1])
                        if count > max_freq:
                            max_freq = count
                        if settings.skip_single_letter_words and len(word) == 1:
                            continue
                        word_freqs[word] = count
                    except Exception as e:
                        pass
    except Exception as e:
        print(f"Error loading word list: {e}")
    return {word: freq/max_freq for word, freq in word_freqs.items()}

def generate_candidate_chords(settings: Settings, layout: config.Layout, word: str) -> Set[FrozenSet[str]]:
    """
    Generate a set of 'meaningful' candidate chords for a word.
    Only include chords that use the letters in the word.
    Include the 'DUP' character if the word has duplicate characters.
    """
    
    # Build a set of available characters
    letters = set(word).union(set(settings.extra_chord_keys))

    # If the word has sufficiently few letters, add some alternate mappings
    if len(letters) <= 4:
        # Add mirror letters
        for letter in letters:
            press = layout.get_from_key(letter)
            if press:
                mirrored = press.get_mirror()
                key = layout.get_key_from_switch(mirrored)
                if key and key not in letters:
                    letters.add(key)
    
    # Limit chord size for ergonomic reasons
    max_chord_size = min(settings.max_chord_size, len(letters))
    
    # Generate all meaningful combinations
    candidates = set()
    for size in range(settings.min_chord_size, max_chord_size + 1):
        for combo in itertools.combinations(letters, size):
            candidates.add(frozenset(combo))
    
    return candidates

def get_pair_cost(settings: Settings, layout: config.Layout, a: str, b: str) -> Optional[float]:
    a_press = layout.get_from_key(a)
    b_press = layout.get_from_key(b)
    # Base cost to discourage large chords
    cost = settings.heuristic.pairwise_cost

    if a_press is None or b_press is None:
        return None

    if a_press.layer != config.Layer.ALPHA or b_press.layer != config.Layer.ALPHA:
        # If either key is not on the alpha layer, we can't use it in a chord
        return None

    # On different hands, it doesn't matter what keys we're pressing
    if a_press.hand != b_press.hand:
        return cost

    # Can't press multiple keys with the same finger
    if a_press.finger == b_press.finger:
        return None

    # Can only press a key "in" if it's the only key on the hand
    if a_press.direction == config.PressDirection.IN or b_press.direction == config.PressDirection.IN:
        return None

    if a_press.finger != config.Finger.THUMB and b_press.finger != config.Finger.THUMB:
        # Can't press keys in two separate directions unless one is a thumb
        if a_press.direction == b_press.direction.get_opposite():
            return None

        if a_press.direction != b_press.direction:
            cost += settings.heuristic.different_direction_cost

    return cost

def chord_cost(settings: Settings, layout: config.Layout, word: str, chord: FrozenSet[str]) -> Optional[float]:
    """
    Heuristic cost function for a chord.
    Lower is better (easier/faster to type).
    """
    if chord in settings.frozen_unavailable_chords:
        return None

    total_cost = 0
    for a, b in itertools.combinations(chord, 2):
        pair_cost = get_pair_cost(settings, layout, a, b)
        if pair_cost is None:
            return None

        total_cost += pair_cost

    # If the first letter of the word is not in the chord, add a penalty
    if word[0] not in chord:
        total_cost += settings.heuristic.missing_first_letter_cost

    if len(word) > 2 and word[1] not in chord:
        # If the second letter of the word is not in the chord, add a penalty
        total_cost += settings.heuristic.missing_second_letter_cost

    # If the last letter of the word is not in the chord, add a penalty
    if len(chord) > 2 and word[-1] not in chord:
        total_cost += settings.heuristic.missing_last_letter_cost

    missing_letters = set(word[2:-1]) - chord
    if len(missing_letters) > 0:
        total_cost += settings.heuristic.missing_any_letter_cost * len(missing_letters)

    # For each letter in the chord, add a penalty if it's not in the word
    for letter in chord:
        if letter != "DUP" and letter not in word:
            total_cost += settings.heuristic.extra_chord_cost if letter in settings.extra_chord_keys else settings.heuristic.bad_letter_cost

    letter_counter = Counter(word)
    duplicate_letters = {letter for letter, count in letter_counter.items() if count > 1}
    if "DUP" in chord and not duplicate_letters:
        # If the chord contains "DUP" but there are no duplicate letters in the word, add a penalty
        total_cost += settings.heuristic.dup_without_duplicates_cost

    if "DUP" in chord and not duplicate_letters.intersection(chord):
        # If the chord contains "DUP" but there are no duplicate letters in the word, add a penalty
        total_cost += settings.heuristic.dup_without_chord_duplicates_cost

    return total_cost

def normalize_cost(cost: float) -> float:
    assert cost >= 0.0 and cost <= 1.0, "Cost must be between 0 and 1"
    # return math.pow(cost, 3)
    return cost

def build_bipartite_graph(
    settings: Settings,
    layout: config.Layout,
    word_list: List[Tuple[str,float]]
) -> Dict[str, Dict[FrozenSet[str], float]]:
    """
    Build a bipartite graph: words ↔ candidate chords.
    Edges are weighted by inverse normalized cost.
    """
    graph = defaultdict(dict)
    max_cost = 0
    for word, freq in tqdm(word_list, desc="Building chord candidate graph", miniters=1000):
        candidates = generate_candidate_chords(settings, layout, word)
        if word in settings.explicit_chords:
            continue
        for chord in candidates:
            # Higher frequency (lower index) and lower cost = higher weight
            cost = chord_cost(settings, layout, word, chord)
            if cost is not None:
                cost = math.pow(freq, settings.heuristic.frequency_exponent) / cost 
                if cost > max_cost:
                    max_cost = cost
                graph[word][chord] = cost

    # Normalize costs to be between 0 and 1
    normalized = defaultdict(dict)
    for word, chord_weights in graph.items():
        for chord, weight in chord_weights.items():
            if max_cost > 0:
                normalized[word][chord] = settings.heuristic.base_weight + normalize_cost(weight / max_cost)
    for word in settings.explicit_chords:
        chord = frozenset(settings.explicit_chords[word])
        # Assign a high weight to explicit chords
        normalized[word][chord] = 1000.0
    return normalized

def maximum_weight_matching(graph: Dict[str, Dict[FrozenSet[str], float]]) -> Dict[str, FrozenSet[str]]:
    """
    Find the optimal assignment of chords to words using networkx's max_weight_matching.
    """
    B = nx.Graph()
    for word, chord_weights in graph.items():
        for chord, weight in chord_weights.items():
            B.add_edge(f"w:{word}", f"c:{','.join(sorted(chord))}", weight=weight, chord=chord, word=word)
    print("Bipartite graph created with nodes:", B.number_of_nodes(), "and edges:", B.number_of_edges())
    print("Calculating maximum weight matching...")
    matching = nx.algorithms.matching.max_weight_matching(B, maxcardinality=False)
    assignments: Dict[str, FrozenSet[str]] = {}
    for u, v in matching:
        if u.startswith("w:") and v.startswith("c:"):
            word = u[2:]
            chord_str = v[2:]
        elif v.startswith("w:") and u.startswith("c:"):
            word = v[2:]
            chord_str = u[2:]
        else:
            continue
        # Find the actual chord object from the graph
        for chord in graph[word]:
            if ','.join(sorted(chord)) == chord_str:
                assignments[word] = chord
                break
    return assignments

class LemmaInfo(NamedTuple):
    word: str
    lemma: str
    modifier: config.Modifier

def get_word_lemmas(settings: Settings, word_freqs: Dict[str, float], layout: config.Layout) -> Dict[str, LemmaInfo]:
    lemmas: Dict[str, LemmaInfo] = {}
    modifiers = layout.get_modifiers()
    for word in tqdm(word_freqs, desc="Finding lemmas for words", miniters=1000):
        best_lemma = None
        for modifier, op in modifiers.items():
            lemma_candidates = op.reverse(word)
            # Find the lemma that has the highest frequency
            for candidate in lemma_candidates:
                if candidate in word_freqs:
                    if best_lemma is None or word_freqs[candidate] > word_freqs[best_lemma.lemma]:
                        best_lemma = LemmaInfo(word=word, lemma=candidate, modifier=modifier)
        if best_lemma is not None:
            lemmas[word] = best_lemma
    return lemmas

def remove_modified_words(settings: Settings, lemmas: Dict[str, LemmaInfo], word_freqs: Dict[str, float]) -> Dict[str, float]:
    """
    Remove words that are modified versions of other words.
    This is to avoid assigning chords to both the original and modified versions.
    """
    result = defaultdict(float)
    for word in tqdm(word_freqs.keys(), desc="Removing words available via modifiers", miniters=1000):
        lemma = lemmas.get(word)
        use_lemma = True
        if lemma is None:
            use_lemma = False
        elif lemma.lemma not in word_freqs:
            use_lemma = False
        elif word_freqs[lemma.lemma] < word_freqs[word] / settings.heuristic.modifier_relative_frequency:
            # If the lemma's frequency is significantly lower than the word's, don't use it
            use_lemma = False

        if use_lemma:
            assert lemma is not None
            result[lemma.lemma] += word_freqs[word]
        else:
            result[word] += word_freqs[word]
    return result

def update_word_freqs(
    settings: Settings,
    word_freqs: Dict[str, float],
    categorizer: Categorizer,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for word, freq in tqdm(word_freqs.items(), desc="Updating word frequencies", miniters=1000):
        updated_freq = freq
        for category in settings.categories:
            mss = categorizer.get_mss(word, category.examples) - categorizer.get_mss(word, category.counterexamples)
            if mss > category.threshold:
                # print(f"Categorizing '{word}' as '{category.name}' with mss {mss:.4f}")
                updated_freq = updated_freq * category.freq_multiplier + category.freq_offset
        if len(word) == 2 and updated_freq < settings.two_word_freq_threshold:
            # If the word is a two-letter word and its frequency is below the threshold, skip it
            continue
        if updated_freq > 0:
            result[word] = updated_freq

    return result

def create_chords(settings: Settings = Settings()) -> config.ChordList:
    original_freqs = load_word_list(settings, "./dataset/unigram_freq.csv")
    word_freqs = update_word_freqs(settings, original_freqs, Categorizer())
    layout = config.Layout()
    lemmas = get_word_lemmas(settings, word_freqs, layout)
    reverse_lemmas = defaultdict(set)
    for word, lemma_info in lemmas.items():
        reverse_lemmas[lemma_info.lemma].add(word)
    updated_freqs = remove_modified_words(settings, lemmas, word_freqs)
    word_list = sorted(updated_freqs.items(), key=lambda x: x[1], reverse=True)
    word_list = word_list[:settings.top_n]
    all_words = set(word[0] for word in word_list)
    for word in settings.explicit_chords:
        if word not in all_words:
            word_list.append((word, 1.0))
    graph = build_bipartite_graph(settings, layout, word_list)
    assignments = maximum_weight_matching(graph)
    # Output assignments
    chord_list = config.ChordList()
    for word, freq in word_list:
        if word in assignments:
            chord = assignments[word]
            chord_list.add_chord(chord, tuple(word))
            print(f"{word} ({original_freqs[word]}): {' + '.join(sorted(chord))}")
            for reverse_word in reverse_lemmas[word]:
                lemma_info = lemmas[reverse_word]
                print(f"    {lemma_info.word} ({original_freqs[lemma_info.word]}): {word} + {lemma_info.modifier}")
        else:
            print(f"{word} ({freq}): No assignment found")
    return chord_list

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Intelligently generates chord assignments for words based on a given layout and word frequency list.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    parser_create = subparsers.add_parser("create", help="Generate the chord assignments based on the provided configuration.")
    parser_create.add_argument(
        "settings_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to the YAML configuration file (optional)."
    )
    parser_create.add_argument(
        "--out",
        type=str,
        default="chords.json",
        help="The output file to write the chord assignments to."
    )

    parser_defaults = subparsers.add_parser("defaults", help="Print the default configuration.")

    parser_convert = subparsers.add_parser("convert", help="Convert a raw JSON file to an easy format or visa-versa.")
    parser_convert.add_argument("in_file", type=str, help="Input file to convert.")
    parser_convert.add_argument("out_file", type=str, help="Output file for the converted data.")
    parser_convert.add_argument(
        "--format",
        choices=["raw", "easy", "need"],
        required=True,
        help="Convert from raw JSON to easy format."
    )

    parser_categories = subparsers.add_parser("categories", help="Dump words in each category to debug.")
    parser_categories.add_argument(
        "settings_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to the YAML configuration file (optional)."
    )

    args = parser.parse_args()

    if args.command == "create":
        # Load YAML configuration
        if args.settings_file is None:
            settings = Settings()
        else:
            with open(args.settings_file, "r") as yaml_file:
                config_data = yaml.safe_load(yaml_file)
                settings = Settings(**config_data)

        # Generate chords
        chord_list = create_chords(settings)
        # Write the chord list to the output file
        with open(args.out, "w") as out_file:
            print("Writing to output file:", args.out)
            json.dump(chord_list.to_easy_json(), out_file, sort_keys=False)
    elif args.command == "defaults":
        # Print the default configuration as yaml
        yaml.dump(
            Settings().model_dump(),
            stream=sys.stdout,
            sort_keys=False,
        )
    elif args.command == "convert":
        with open(args.in_file, "r") as in_file:
            data = json.load(in_file)
        if "type" not in data:
            raise ValueError("Input JSON file does not contain 'type' field to determine format.")
        if data["type"] == "chords":
            chord_list = config.ChordList.from_raw_json(data)
        elif data["type"] == "easychords":
            chord_list = config.ChordList.from_easy_json(data)
        else:
            raise ValueError(f"Unknown type '{data['type']}' in input JSON file.")

        if args.format == "need":
            output = chord_list.to_need_json()
        elif args.format == "raw":
            output = chord_list.to_raw_json()
        elif args.format == "easy":
            output = chord_list.to_easy_json()
        else:
            raise ValueError(f"Unknown format '{args.format}' specified for conversion.")
            
        with open(args.out_file, "w") as out_file:
            json.dump(output, out_file, sort_keys=False)
    elif args.command == "categories":
        # Load YAML configuration
        if args.settings_file is None:
            settings = Settings()
        else:
            with open(args.settings_file, "r") as yaml_file:
                config_data = yaml.safe_load(yaml_file)
                settings = Settings(**config_data)

        categories = []
        word_list = load_word_list(settings, "./dataset/unigram_freq.csv")
        ordered_word_list = sorted(word_list.items(), key=lambda x: x[1], reverse=True)
        categorizer = Categorizer()
        for category in settings.categories:
            info = {"name": category.name, "words": []}
            for word, freq in tqdm(ordered_word_list, desc=f"Category '{category.name}'", miniters=1000):
                mss = categorizer.get_mss(word, category.examples) - categorizer.get_mss(word, category.counterexamples)
                if mss > category.threshold:
                    info["words"].append({
                        "word": word,
                        "mss": mss,
                        "freq": freq,
                    })
            categories.append(info)
        print(yaml.dump(categories, sort_keys=False))
    else:
        parser.print_help()


