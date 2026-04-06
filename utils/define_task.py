"""
    Automatically define task instructions from environment initial observations and goals, using a large language model (LLM).
"""

def generate_robot_instruction(file_path: str) -> str:
    """
    Parses a environment spec file and returns a formatted task string.
    Ignores any lines that don't match the required objects.

    Uses a standard instruction format: "place the <pick_object> in the <place_object>". 
    See functions below for semantic language variations

    # Usage
    # instruction = generate_robot_instruction('env_spec.txt')
    # print(instruction)
    """
    pick_obj = None
    place_obj = None
    directions = {"top", "bottom", "left", "right", "middle", "center"}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Clean up whitespace and split by colon
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip().lower()
                    
                    # Specifically look for our targets
                    if 'pick object'==key:
                        pick_obj = value
                    elif 'place object'==key and place_obj is None:  # Only set place_obj if it hasn't been set yet
                        place_obj = value
        if pick_obj and place_obj:
            return f"place the {pick_obj} in the {place_obj}"
        else:
            return "Error: Could not find both pick and place objects in the file."
            
    except FileNotFoundError:
        return "Error: Specification file not found."


