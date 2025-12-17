"""
Predefined lists of visual attribute words for analyzing nearest neighbors.

These word lists serve as proxies for concreteness and visual grounding.
"""

# Color words - comprehensive list including basic colors, shades, and variations
# Based on common color naming conventions and CSS color names
COLORS = [
    # Basic colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey",
    
    # Red spectrum
    "crimson", "scarlet", "ruby", "burgundy", "maroon", "cherry", "rose",
    "salmon", "coral", "brick", "rust", "wine", "vermillion",
    
    # Blue spectrum
    "navy", "azure", "cobalt", "sapphire", "turquoise", "cyan", "teal",
    "indigo", "periwinkle", "sky", "powder",
    
    # Green spectrum
    "lime", "olive", "emerald", "jade", "mint", "forest", "sage",
    "chartreuse", "seafoam", "moss", "grass",
    
    # Yellow/Orange spectrum
    "gold", "golden", "amber", "ochre", "mustard", "lemon", "cream",
    "peach", "apricot", "tangerine", "copper", "bronze",
    
    # Purple/Pink spectrum
    "violet", "lavender", "magenta", "fuchsia", "plum", "lilac", "mauve",
    "orchid", "blush", "hot",
    
    # Brown spectrum
    "tan", "beige", "khaki", "chocolate", "coffee", "mahogany", "sienna",
    "umber", "taupe", "sand", "caramel", "chestnut",
    
    # Grayscale
    "silver", "charcoal", "slate", "ash", "smoke", "pearl", "ivory",
    "ebony", "jet", "snow",
    
    # Color descriptors
    "bright", "dark", "light", "pale", "deep", "vivid", "pastel",
    "neon", "metallic", "matte", "glossy", "shiny",
]

# Shape words - geometric and organic shapes
SHAPES = [
    # Basic 2D shapes
    "round", "circular", "square", "rectangular", "triangular", "oval",
    "elliptical", "hexagonal", "octagonal", "pentagonal",
    
    # 3D shapes
    "spherical", "cubic", "cylindrical", "conical", "pyramidal",
    "spherical", "tubular", "boxy",
    
    # Curves and lines
    "curved", "straight", "angular", "rounded", "pointed", "sharp",
    "wavy", "zigzag", "spiral", "helical", "twisted", "coiled",
    
    # Shape descriptors
    "flat", "hollow", "solid", "thin", "thick", "wide", "narrow",
    "broad", "elongated", "stretched", "compressed", "bulbous",
    
    # Organic shapes
    # "irregular", "organic", "geometric", "symmetrical", "asymmetrical",
    # "jagged", "smooth", "ragged", "tapered", "flared",
    
    # Size-related (often co-occur with shapes)
    "large", "small", "tiny", "huge", "massive", "gigantic", "minuscule",
]

# Texture words - surface qualities
TEXTURES = [
    # Tactile textures
    "smooth", "rough", "bumpy", "coarse", "fine", "soft", "hard",
    "slippery", "sticky", "slimy", "gritty", "grainy", "sandy",
    "silky", "velvety", "fuzzy", "fluffy", "furry", "hairy",
    
    # Visual textures
    "shiny", "glossy", "matte", "dull", "reflective", "transparent",
    "opaque", "translucent", "clear", "cloudy", "hazy", "misty",
    
    # Material-like textures
    "metallic", "wooden", "leathery", "fabric", "plastic", "glassy",
    "crystalline", "papery", "rubbery", "waxy", "oily", "greasy",
    
    # Surface patterns
    "striped", "spotted", "dotted", "checkered", "patterned", "plain",
    "textured", "embossed", "engraved", "carved", "etched",
    
    # Condition/wear
    # "worn", "weathered", "aged", "new", "pristine", "scratched",
    # "dented", "cracked", "chipped", "polished", "rusty", "tarnished",
    
    # Natural textures
    "rocky", "stony", "pebbly", "muddy", "grassy", "leafy", "bark",
    "scaly", "feathery", "mossy", "icy", "snowy", "frosty",
]

# Combined dictionary for easy access
VISUAL_ATTRIBUTES = {
    'color': COLORS,
    'shape': SHAPES,
    'texture': TEXTURES,
}

# All words in a single set for quick lookup
ALL_VISUAL_WORDS = set(COLORS + SHAPES + TEXTURES)


def get_attribute_type(word):
    """Return the attribute type(s) for a given word."""
    word_lower = word.lower().strip()
    types = []
    if word_lower in COLORS:
        types.append('color')
    if word_lower in SHAPES:
        types.append('shape')
    if word_lower in TEXTURES:
        types.append('texture')
    return types


def is_visual_attribute(word):
    """Check if a word is a visual attribute."""
    return word.lower().strip() in ALL_VISUAL_WORDS

