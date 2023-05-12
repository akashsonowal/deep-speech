import json

char_map_str = json.load("char_map.json")

class TextTransform:
  """Maps characters to integers and vice versa"""
  def __init__(self):
    self.char_map_str = char_map_string
    
  
