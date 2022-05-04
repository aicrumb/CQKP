# CQKP - Contrastive Question-Knowledge Pretraining

---

I built this so I can have a sort of automatic retrieval method for finding the best article for a question, but it can be used both ways if you're into that?

### "setup"

you can do something like 

```bash
wget https://raw.githubusercontent.com/aicrumb/CQKP/master/cqkp.py
```

or just any way to get the script

and requirements (omit: pytorch, because setup for pytorch is different for every circumstance):

```bash
pip install transformers wandb
```

### use

picking the best article from a question

```python
import cqkp
model = cqkp.load_model(download=True) # you'll have to sign into wandb the first time
articles = [
    "Birds can eat all nuts other than the usual peanuts",
    "Birds are a group of warm-blooded vertebrates constituting the class Aves",
]
model.best_answer("what is a bird?", articles)[0]
# expected output:
# "Birds are a group of warm-blooded vertebrates constituting the class Aves"
```

picking the best question from an article

```python
questions = [
    "what is a bird?",
    "what nuts can birds eat?"
]
answer = "Birds can eat all nuts other than the usual peanuts"
model.best_question(questions,answer)[0]
# expected output:
# what nuts can birds eat?
```

both of those commands (best_answer, best_question) return the best match, the index of the best match in the list you gave it, and the score all in a tuple

for training:

theres a notebook in the repo that requires squad in csv form (https://www.kaggle.com/datasets/ananthu017/squad-csv-format)



\- *crumb*