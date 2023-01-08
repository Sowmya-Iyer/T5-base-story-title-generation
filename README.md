# T5-base-story-title-generation

Dataset comprises of Movies and TV shows description and titles from the following OTT platforms:
- Netflix 
- Amazon prime 
- Hulu
- DisneyPlus

This has been deployed in hugging face in the following link:
[Link to hugging face API](https://huggingface.co/NeuralNerd/t5-base-story-title-generation)

Can be directly imprted from the transformers library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("NeuralNerd/t5-base-story-title-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("NeuralNerd/t5-base-story-title-generation")

#input_max_length=500
text = """
Puss in Boots discovers that his passion for adventure has taken its toll: he has burnt through eight of his nine lives. Puss sets out on an epic journey to find the mythical Last Wish and restore his nine lives.
"""

inputs = ["summarize: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

print(decoded_output)
```
