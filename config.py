from enum import Enum
from typing import Dict, NamedTuple, Optional, NewType, cast, Tuple, List, FrozenSet
import csv
import json
import re
from datetime import datetime

KeyCode = NewType('KeyCode', int)
SwitchCode = NewType('SwitchCode', int)

class PressDirection(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    IN = "in"

    def get_opposite(self):
        """Get the opposite direction."""
        opposites = {
            PressDirection.UP: PressDirection.DOWN,
            PressDirection.DOWN: PressDirection.UP,
            PressDirection.LEFT: PressDirection.RIGHT,
            PressDirection.RIGHT: PressDirection.LEFT,
            PressDirection.IN: PressDirection.IN,  # IN has no opposite
        }
        return opposites[self]

class Finger(Enum):
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
    THUMB = "thumb"
    PALM = "palm"
    OTHER = "other"  # For any other finger not explicitly defined

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class Layer(Enum):
    ALPHA = "alpha"
    NUMERIC = "numeric"
    FUNCTIONAL = "functional"
    NONE = "none"  # Represents no layer or default state

class Modifier(Enum):
    NONE = "none"
    PLURAL = "plural"
    COMPARATIVE = "comparative"
    PRESENT_PARTICIPLE = "present_participle"
    PAST = "past"

class SwitchPress(NamedTuple):
    code: SwitchCode
    finger: Finger
    hand: Hand
    direction: PressDirection
    layer: Layer = Layer.NONE

    def get_mirror(self) -> 'SwitchPress':
        """Get the mirror of this SwitchPress."""
        mirror_hand = Hand.RIGHT if self.hand == Hand.LEFT else Hand.LEFT
        mirror_direction = self.direction.get_opposite() if self.direction in (PressDirection.LEFT, PressDirection.RIGHT) else self.direction
        return SwitchPress(
            code=self.code,
            finger=self.finger,
            hand=mirror_hand,
            direction=mirror_direction,
            layer=self.layer
        )

def load_key_codes(path: str) -> Dict[str, KeyCode]:
    """Load key codes from a CSV file with key,code format."""
    key_codes = {}
    delimiter = '\t' if path.endswith('.tsv') else ','
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(
                f,
                delimiter=delimiter,
                quoting=csv.QUOTE_NONE,
            )
            for row in reader:
                if len(row) == 6:
                    try:
                        code, hex, category, key, description, notes = row
                        key_codes[key] = int(code)
                    finally:
                        continue
    except Exception as e:
        raise ValueError(f"Error loading key codes from {path}: {e}")
    return key_codes

KEY_CODES = load_key_codes("./assets/CCOSActionCodes.tsv")
KEY_CODES_REVERSE = {v: k for k, v in KEY_CODES.items() if v != ""}

class ModifierMapping(NamedTuple):
    pre: str
    post: str
    pre_re: re.Pattern
    post_re: re.Pattern
    reverse: str

mod_pattern = re.compile(r'^[*!.=]*')

class ModifierOperation:
    def __init__(self, modifier_csv: str, default_key: str):
        self.default_key = default_key
        self._modifiers: List[ModifierMapping] = []
        try:
            with open(modifier_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        pre, post = row
                        pre_re = re.compile("^" + mod_pattern.sub(lambda m: f"({m.group(0)})", pre).replace("=", "[aeiou]").replace("!", "[^aeiou]").replace("*", ".*") + "$")
                        post_re = re.compile("^(.*)" + post + "$")
                        raw_pre = re.match(r'^[^a-z ]*([a-z ]*)$', pre)
                        assert raw_pre is not None, f"Huh? Regex should always match: {pre}"
                        reverse = raw_pre.group(1) if raw_pre else ""
                        self._modifiers.append(ModifierMapping(
                            pre=pre,
                            post=post,
                            pre_re=pre_re,
                            post_re=post_re,
                            reverse=reverse
                        ))
        except Exception as e:
            raise ValueError(f"Error loading modifier CSV file: {e}")
        
    def apply(self, word: str) -> str:
        """Apply modifiers to a word."""
        for modifier in self._modifiers:
            match = modifier.pre_re.match(word)
            if match:
                return match.group(1) + modifier.post
        return word

    def reverse(self, word: str) -> List[str]:
        """Reverse the modifiers applied to a word."""
        reverses = []
        for modifier in self._modifiers:
            match = modifier.post_re.match(word)
            if match:
                 candidate = match.group(1) + modifier.reverse
                 if self.apply(candidate) == word:
                    reverses.append(candidate)
        return reverses

class Layout:
    def __init__(self, layout_file: str = "./state/layout.json"):
        try:
            with open(layout_file, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading layout json file: {e}")
        if not isinstance(layout_data, dict):
            raise ValueError("Layout file must be a dictionary.")
        if layout_data.get("type") != "layout":
            raise ValueError("Layout file does not appear to be of type layout.")
        if layout_data.get("device") != "ONE":
            raise ValueError("Unsupported device type. Only the CC1 is supported.")
        if layout_data.get("charaVersion") != 1:
            raise ValueError("Unsupported charaVersion. Only version 1 is supported.")
        self._switch_codes: Dict[SwitchCode, SwitchPress] = {}
        self._switch_codes_reverse: Dict[SwitchPress, SwitchCode] = {}
        self._bindings: Dict[Tuple[Layer,SwitchCode], KeyCode] = {}
        self._bindings_reverse: Dict[KeyCode, Tuple[Layer, SwitchCode]] = {}
        self._max_code = 0

        # Left Hand
        self.__add_finger(0, Hand.LEFT, Finger.PALM)
        self.__add_finger(5, Hand.LEFT, Finger.THUMB)
        self.__add_finger(10, Hand.LEFT, Finger.THUMB)
        self.__add_finger(15, Hand.LEFT, Finger.INDEX)
        self.__add_finger(20, Hand.LEFT, Finger.MIDDLE)
        self.__add_finger(25, Hand.LEFT, Finger.RING)
        self.__add_finger(30, Hand.LEFT, Finger.PINKY)
        self.__add_finger(35, Hand.LEFT, Finger.OTHER)
        self.__add_finger(40, Hand.LEFT, Finger.OTHER)

        # Right Hand
        self.__add_finger(45, Hand.RIGHT, Finger.PALM)
        self.__add_finger(50, Hand.RIGHT, Finger.THUMB)
        self.__add_finger(55, Hand.RIGHT, Finger.THUMB)
        self.__add_finger(60, Hand.RIGHT, Finger.INDEX)
        self.__add_finger(65, Hand.RIGHT, Finger.MIDDLE)
        self.__add_finger(70, Hand.RIGHT, Finger.RING)
        self.__add_finger(75, Hand.RIGHT, Finger.PINKY)
        self.__add_finger(80, Hand.LEFT, Finger.OTHER)
        self.__add_finger(85, Hand.LEFT, Finger.OTHER)

        # Load the layout bindings
        self._load_bindings(layout_data["layout"][2], Layer.FUNCTIONAL)
        self._load_bindings(layout_data["layout"][1], Layer.NUMERIC)
        self._load_bindings(layout_data["layout"][0], Layer.ALPHA)

        self.modifiers: Dict[Modifier, ModifierOperation] = {
            Modifier.PLURAL: ModifierOperation("./assets/plural.csv", "AMBIRIGHT"),
            Modifier.COMPARATIVE: ModifierOperation("./assets/comparative.csv", "KM_2_R"),
            Modifier.PRESENT_PARTICIPLE: ModifierOperation("./assets/presentParticiple.csv", "AMBILEFT"),
            Modifier.PAST: ModifierOperation("./assets/past.csv", "KM_2_L"),
        }

    def get_modifiers(self) -> Dict[Modifier, ModifierOperation]:
        """Get the modifier operations."""
        return self.modifiers

    def __add_key(self, code: SwitchCode, hand: Hand, finger: Finger, direction: PressDirection):
        key_press = SwitchPress(code=code, hand=hand, finger=finger, direction=direction)
        self._switch_codes[code] = key_press
        self._switch_codes_reverse[key_press] = code
        self._max_code = max(self._max_code, code)

    def __add_finger(self, base_code: int, hand: Hand, finger: Finger):
        if hand == Hand.LEFT:
            self.__add_key(cast(SwitchCode, base_code), hand, finger, PressDirection.IN)
            self.__add_key(cast(SwitchCode, base_code + 1), hand, finger, PressDirection.RIGHT)
            self.__add_key(cast(SwitchCode, base_code + 2), hand, finger, PressDirection.UP)
            self.__add_key(cast(SwitchCode, base_code + 3), hand, finger, PressDirection.LEFT)
            self.__add_key(cast(SwitchCode, base_code + 4), hand, finger, PressDirection.DOWN)
        else:
            self.__add_key(cast(SwitchCode, base_code), hand, finger, PressDirection.IN)
            self.__add_key(cast(SwitchCode, base_code + 1), hand, finger, PressDirection.LEFT)
            self.__add_key(cast(SwitchCode, base_code + 2), hand, finger, PressDirection.UP)
            self.__add_key(cast(SwitchCode, base_code + 3), hand, finger, PressDirection.RIGHT)
            self.__add_key(cast(SwitchCode, base_code + 4), hand, finger, PressDirection.DOWN)

    def _load_bindings(self, layout: List[KeyCode], layer: Layer):
        for i in range(0, len(layout)):
            switch_code = cast(SwitchCode,i)
            key_code = layout[i]
            self._bindings[(layer, switch_code)] = key_code
            self._bindings_reverse[key_code] = (layer, switch_code)

    def get_from_switch_code(self, code: SwitchCode) -> Optional[SwitchPress]:
        """Get SwitchPress object by code."""
        return self._switch_codes.get(code)

    def get_from_key(self, key: str) -> Optional[SwitchPress]:
        """Get SwitchPress object by key name."""
        code = KEY_CODES.get(key)
        if code is None:
            return None
        layer, switch_code = self._bindings_reverse.get(code, (Layer.NONE, None))
        if switch_code is None:
            return None
        press = self.get_from_switch_code(switch_code)
        if press is None:
            return None
        return press._replace(layer=layer)

    def get_key_from_switch(self, switch: SwitchPress) -> Optional[str]:
        key_code = self._bindings.get((switch.layer, switch.code))
        if key_code is None:
            return None
        return KEY_CODES_REVERSE.get(key_code, None)

class Chord(NamedTuple):
    input: FrozenSet[str]
    output: Tuple[str, ...]

class ChordList:
    def __init__(self):
        self._chords: Dict[FrozenSet[str],Chord] = {}

    def add_chord(self, input_keys: FrozenSet[str], output_keys: Tuple[str, ...], overwrite: bool = False) -> None:
        # Make sure we don't already have this chord
        if input_keys in self._chords:
            if not overwrite:
                raise ValueError("Chord with the same input already exists.")
            del self._chords[input_keys]
        # Add the new chord
        self._chords[input_keys] = Chord(input=input_keys, output=output_keys)

    def get_chord(self, input_keys: FrozenSet[str]) -> Optional[Chord]:
        """Get a chord by its input keys."""
        return self._chords.get(input_keys)

    @classmethod
    def from_raw_json(cls, data: dict) -> 'ChordList':
        """Create a ChordList from raw JSON data."""
        assert "charaVersion" in data and data["charaVersion"] == 1, "Unsupported charaVersion. Only version 1 is supported."
        assert "type" in data and data["type"] == "chords", "Data must be of type 'chords'."
        assert "chords" in data and type(data["chords"]) is list, "Data must contain a 'chords' list."
        chords = data["chords"]
        chord_list = cls()
        for item in chords:
            inputs = frozenset([KEY_CODES_REVERSE[code] for code in item[0] if code != 0])
            outputs = tuple(KEY_CODES_REVERSE[code] for code in item[1] if code != 0)
            chord_list._chords[inputs] = Chord(input=inputs, output=outputs)
        return chord_list

    def to_raw_json(self) -> dict:
        """Convert the ChordList to raw JSON data."""
        chords = []
        for chord in self._chords.values():
            input_codes = [KEY_CODES[key] for key in chord.input]
            if len(input_codes) < 12:
                input_codes = [0] * (12 - len(input_codes)) + input_codes
            input_codes = sorted(input_codes)
            output_codes = [KEY_CODES[key] for key in chord.output]
            chords.append([input_codes, output_codes])
        return {
            "charaVersion": 1,
            "type": "chords",
            "chords": chords
        }

    @classmethod
    def from_easy_json(cls, data: dict) -> 'ChordList':
        assert "type" in data and data["type"] == "easychords", "Data must be of type 'easychords'."
        assert "chords" in data and type(data["chords"]) is list, "Data must contain a 'chords' list."
        chords = data["chords"]
        chord_list = cls()
        for item in chords:
            inputs = frozenset(item[0])
            outputs = tuple(item[1])
            chord_list._chords[inputs] = Chord(input=inputs, output=outputs)
        return chord_list

    def to_easy_json(self) -> dict:
        """Convert the ChordList to easy JSON data."""
        chords = []
        for chord in self._chords.values():
            chords.append([sorted(list(chord.input)), list(chord.output)])
        return {
            "type": "easychords",
            "chords": chords
        }

    def to_need_json(self) -> dict:
        """Convert the ChordList to need JSON data."""
        chords = {}
        now = datetime.utcnow().isoformat() + "Z"
        for chord in self._chords.values():
            if any(len(char) > 1 for char in chord.output):
                continue
            output = "".join(chord.output)
            chords[output] = {
                "Key": output,
                "Value": " + ".join(sorted(list(chord.input))),
                "Created": now,
            }
        return chords

