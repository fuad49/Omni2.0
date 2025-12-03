import google.generativeai as genai
from PIL import Image
from io import BytesIO
import logging
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("OmniVision")

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

def process_image(image_bytes: bytes):
    """
    Runs the AI pipeline using Google Gemini:
    1. Gemini 1.5 Flash: Describes the product in the image.
    2. Gemini Text Embedding: Embeds the description (768 dims).
    """
    try:
        # 1. Load Image
        image = Image.open(BytesIO(image_bytes))
        
        # 2. Generate Description (The Eye & The Judge)
        # We ask Gemini to describe the main product in detail, focusing on visual features.
        # Using gemini-2.0-flash as it is available in user's account
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = "Describe the main product in this image in detail. Focus on category, color, material, pattern, and style. Be concise but specific."
        
        response = model.generate_content([prompt, image])
        description = response.text
        logger.info(f"Gemini Description: {description}")
        
        # 3. Generate Embedding (The Scout)
        # We embed the description to get a semantic vector.
        # Model: models/text-embedding-004 (768 dimensions)
        emb_result = genai.embed_content(
            model="models/text-embedding-004",
            content=description,
            task_type="retrieval_document",
            title="Product Description"
        )
        embedding = emb_result['embedding']
        
        # 4. Return Result
        # We use the same embedding for both fields to maintain compatibility with the dual-vector schema structure
        # (or we could simplify the schema, but let's just duplicate for now to minimize code changes in main.py)
        return {
            "siglip_embedding": embedding, # 768 dims
            "dino_embedding": embedding,   # 768 dims (Duplicate)
            "description": description     # Optional: Save this if we want
        }

    except Exception as e:
        logger.error(f"Gemini AI Error: {e}")
        # Return zero vectors on failure
        return {
            "siglip_embedding": [0.0] * 768,
            "dino_embedding": [0.0] * 768
        }

def load_models():
    pass

def verify_visual_match(image1_bytes: bytes, image2_bytes: bytes) -> int:
    """
    Compares two images using Gemini and returns a similarity score (0-100).
    """
    try:
        img1 = Image.open(BytesIO(image1_bytes))
        img2 = Image.open(BytesIO(image2_bytes))
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """
        Compare these two images. Are they the EXACT same product model?
        Ignore lighting, angle, or minor wear.
        Focus on:
        1. Shape (Round vs Square)
        2. Dial details (Markers, Hands, Sub-dials)
        3. Strap style
        
        Return ONLY a number from 0 to 100 representing the probability they are the same product.
        0 = Different product.
        100 = Exact same product.
        """
        
        response = model.generate_content([prompt, img1, img2])
        text = response.text.strip()
        
        # Extract number
        import re
        match = re.search(r'\d+', text)
        if match:
            score = int(match.group())
            logger.info(f"Visual Verification Score: {score}")
            return score
        return 0
    except Exception as e:
        logger.error(f"Visual Verification Error: {e}")
        return 0
