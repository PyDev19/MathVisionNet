from xml.etree import ElementTree

def parse_inkml(file_path: str) -> tuple:
    """Parses inkml file to return the stroke sequences and LaTeX of the equation within the inkml

    Args:
        file_path (str): Path to the inkml file
    
    Returns:
        tuple: A tuple containing the stroke sequences and LaTeX of the equation
    """
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    
    # Extract the LaTeX representation of the equation
    latex = None
    for annotation in root.findall(".//{http://www.w3.org/2003/InkML}annotation"):
        if annotation.attrib.get("type") == "normalizedLabel":
            latex = annotation.text
            break
    
    # Extract the stroke sequences
    strokes = []
    for trace in root.findall(".//{http://www.w3.org/2003/InkML}trace"):
        stroke = []
        if not trace.text:
            continue
        
        for point in trace.text.strip().split(","):
            x, y, t = map(float, point.split())
            stroke.append((x, y, t))
        strokes.append(stroke)
    
    return strokes, latex