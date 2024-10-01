import re
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the custom NER model (assuming it's in the ner_model folder)
model_path = "ner_model"
nlp_custom = spacy.load(model_path)

# Load spaCy's small model (pre-trained) to extract additional person names
nlp_sm = spacy.load("en_core_web_sm")

# Text request body
class TextRequest(BaseModel):
    text: str

# Clean the input text
def clean_text(text: str) -> str:
    """Normalize whitespaces and remove excess newlines."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Check for restricted words
def is_near_restricted_words(ent, text, restricted_words=["Copyright", "Published"]):
    """Check if an entity is near words like 'Copyright' or 'Published'."""
    context_range = 20
    start_range = max(0, ent.start_char - context_range)
    end_range = min(len(text), ent.end_char + context_range)
    context = text[start_range:end_range].lower()
    return any(word.lower() in context for word in restricted_words)

# Endpoint to process text and return named entities
@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    cleaned_text = clean_text(text)

    # Process with custom NER model to extract the first author and other entities
    doc_custom = nlp_custom(cleaned_text)
    entities_to_store = []
    first_author = None

    # Process custom NER entities (Title, Edition, Volume, ISBN, First Author)
    for ent in doc_custom.ents:
        # Capture all entities (Title, Edition, Volume, ISBN, Authors)
        if ent.label_ == "ISBN":
            entities_to_store.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        elif ent.label_ == "Authors" and first_author is None:
            # Capture the first author only
            entities_to_store.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            first_author = ent.text  # Save the first author name to avoid duplicates
        elif ent.label_ not in ["Authors"]:
            # Capture other entities like Title, Edition, Volume, etc.
            entities_to_store.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

    # Process with spaCy small model to extract additional authors (starting from the second author)
    doc_sm = nlp_sm(cleaned_text)
    for ent in doc_sm.ents:
        # Detect additional authors, but avoid the first author
        if ent.label_ == "PERSON" and ent.text != first_author and not is_near_restricted_words(ent, cleaned_text):
            entities_to_store.append({
                "text": ent.text,
                "label": "Authors",
                "start": ent.start_char,
                "end": ent.end_char
            })

    # Return the entities in the dictionary format
    return {"entities": entities_to_store}

# Health check route
@app.get("/")
async def health_check():
    return {"status": "Model is running"}
