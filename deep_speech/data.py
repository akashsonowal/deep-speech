import json

class TextTransform:
  """Maps characters to integers and vice versa"""
  def __init__(self):
    self.char_map_str = json.load(open("char_map.json", "r"))
    self.char_map = {}
    self.index_map = {}
    for line in char_map_str.strip().split("\n"):
      ch, index = line.split()
      self.char_map[ch] = int(index)
      self.index_map[int(index)] = ch
    self.index_map[1] = ' '
   
  def text_to_int(self, text):
    """Use a character map and convert text to an integer sequence"""
    int_sequence = []
    for c in text:
      if c == ' ':
        ch = self.char_map['']
      else:
        ch = self.char_map[c]
    int_sequence.append(ch)
    
        
    
  
