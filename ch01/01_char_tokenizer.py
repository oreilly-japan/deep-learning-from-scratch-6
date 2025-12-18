text = "hello世界😁"
print(list(text))  # ['h', 'e', 'l', 'l', 'o', '世', '界', '😁']

print(ord('h'))  # 104
print(ord('😁'))  # 128513

print(chr(104))    # 'h'
print(chr(128513)) # '😁'

ids = [ord(w) for w in list(text)]
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello世界😁"

# エンコード
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

# デコード
decoded = tokenizer.decode(ids)
print(decoded)  # hello世界😁