import re

def get_ansi_to_html_color(ansi_code):
    # Mapping ANSI colors to HTML colors
    ansi_to_html = {
        208: (255, 127, 0),    # Bright Orange (HTML)
        227: (255, 255, 0),    # Bright Yellow (HTML)
        71:  (204, 204, 0),    # Lighter Yellow (HTML)
        114: (178, 178, 0),    # Yellow-Green (HTML)
        160: (153, 204, 0),    # Medium Yellow-Green (HTML)
        215: (102, 255, 0),    # Bright Green (HTML)
        166: (0, 255, 0)       # Pure Green (HTML)
    }

    # Default color in case the ANSI code isn't in the map
    color = ansi_to_html.get(ansi_code, (255, 255, 255))  # White as fallback
    return f"rgb({color[0]}, {color[1]}, {color[2]})"

def convert_ansi_to_html(text):
    ansi_escape = re.compile(r'(\033\[38;5;(\d+)m)(.*?)\033\[0m')

    def replace_color(match):
        ansi_code = int(match.group(2))
        color = get_ansi_to_html_color(ansi_code)
        content = match.group(3)
        return f'<span style="color: {color};">{content}</span>'

    return ansi_escape.sub(replace_color, text)