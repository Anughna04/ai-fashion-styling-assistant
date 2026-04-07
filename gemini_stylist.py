"""
Gemini AI Stylist - Generates personalized fashion recommendations
"""

from google import genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

class GeminiStylist:
    """AI Fashion Stylist using Gemini"""
    
    def __init__(self):
        # We fetch explicitly after load_dotenv ensures variables are in os.environ
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_actual_gemini_api_key_here":
            self.client = genai.Client(api_key=api_key)
            self.available = True
            print("✅ Gemini AI Stylist initialized")
        else:
            self.available = False
            print("⚠️ Gemini API key not found - using fallback stylist")
    
    def generate_detailed_advice(self, items, user_query, occasion, season, image_description=None):
        """Generate comprehensive styling advice"""
        
        if not self.available:
            return self._fallback_advice(items, occasion, season)
        
        # Prepare item details
        items_detail = []
        for idx, item in enumerate(items[:6], 1):
            items_detail.append(f"{idx}. {item.get('name', 'Fashion Item')} - "
                              f"Category: {item.get('category', 'N/A')}, "
                              f"Style: {item.get('style', 'N/A')}")
        
        items_text = "\n".join(items_detail)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert fashion stylist with 20 years of experience. A client has come to you with the following:

CLIENT REQUEST: "{user_query}"
OCCASION: {occasion or 'Not specified - provide versatile options'}
SEASON: {season or 'All seasons'}
{'IMAGE UPLOADED: ' + image_description if image_description else 'NO IMAGE PROVIDED'}

AVAILABLE FASHION ITEMS TO WORK WITH:
{items_text}

Provide detailed, personalized styling advice in the following structure:

## 🎨 COMPLETE OUTFIT COMBINATIONS

Create 2-3 distinct outfit combinations using the available items. For each combination:
- Specify exact items to combine
- Explain why this combination works
- Suggest layering if applicable

## 💎 COLOR COORDINATION MASTERY

- Analyze the color palette of selected items
- Explain color theory principles at play
- Suggest complementary accent colors
- Provide skin tone considerations (warm/cool undertones)
- Recommend color blocking or monochrome techniques

## ✨ JEWELRY & ACCESSORIES STRATEGY

Be VERY specific:
- **Necklace**: Type (pendant/choker/statement), length, metal (gold/silver/rose gold)
- **Earrings**: Style (studs/hoops/drops), size (delicate/statement)
- **Bracelets**: Style and stacking suggestions
- **Rings**: Minimalist vs statement pieces
- **Watches**: Style and when to wear
- **Bags**: Size, style, color - day bag vs evening clutch
- **Belts**: Width, color, when to add
- **Scarves/Shawls**: How to style them

## 👠 FOOTWEAR EXPERTISE

Provide specific recommendations:
- Exact shoe styles (pumps/loafers/ankle boots/sandals)
- Heel heights (flat/2"/3"/4"+) and when appropriate
- Color matching rules
- Comfort vs style balance
- Alternative options for different formality levels

## 🧵 FABRIC & TEXTURE INTELLIGENCE

- How to mix different fabric weights
- Texture combinations (smooth silk + textured wool)
- Seasonal fabric appropriateness
- Layering strategies for temperature control
- Care instructions for delicate pieces

## 📐 PROPORTION & SILHOUETTE GUIDANCE

- Body shape considerations
- Balance loose and fitted pieces
- Tucking, half-tuck, or leaving untucked
- Belt placement and when to use
- Hemline and proportion rules
- How to create elongating effects

## 💄 BEAUTY & GROOMING COORDINATION

- Hair styling suggestions for each outfit
- Makeup color palette recommendations
- Nail polish color ideas
- Perfume notes that complement the style

## 📈 CURRENT TRENDS vs TIMELESS STYLE

- Which current trends apply to this look
- Timeless elements that never go out of style
- How to make trends work without looking dated
- Investment pieces worth the splurge

## 🛍️ SHOPPING & STYLING HACKS

- Where to find similar items (high-end and budget-friendly)
- Which pieces are worth investing in
- Mix high and low fashion tips
- How to restyle items for different occasions
Ensure all sentences and bullet points are written as single continuous lines without any line breaks within a sentence or phrase.
## ⚠️ WHAT TO AVOID

- Common styling mistakes for this occasion
- Colors/patterns that clash
- Over-accessorizing warnings
- Inappropriate elements for the context

Be warm, encouraging, and specific. Give exact details, not vague suggestions. Make the client feel confident and excited about their outfit choices!"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return self._fallback_advice(items, occasion, season)
    
    def _fallback_advice(self, items, occasion, season):
        """Fallback advice when Gemini is unavailable"""
        advice = f"""## 🎨 Styling Recommendations

**Note**: This is basic styling advice. Configure your Gemini API key for personalized AI recommendations!

**Found {len(items)} matching items for your search.**

### Outfit Combinations:
Combine items from different categories to create complete looks. Mix tops with bottoms, add outerwear for layers.

### Color Coordination:
- Neutrals (black, white, navy, grey) work with everything
- Add one accent color for interest
- Match metals in jewelry (all gold or all silver)

### Jewelry Selection:
- **Casual**: Minimal jewelry, small studs, delicate chains
- **Formal**: Statement pieces, bold necklaces or earrings (not both)
- **Business**: Simple, elegant pieces

### Footwear:
- **Casual**: Sneakers, flats, ankle boots
- **Business**: Closed-toe heels, loafers, oxford shoes
- **Formal**: Classic pumps, strappy heels
- **Party**: Bold heels, metallic finishes

### Seasonal Tips for {season or 'All Seasons'}:
"""
        
        if season == "Summer":
            advice += "Light fabrics, bright colors, breathable materials, sandals and open-toe shoes"
        elif season == "Winter":
            advice += "Layer pieces, warm fabrics, boots, darker rich colors, add scarves"
        elif season == "Spring":
            advice += "Pastels, floral patterns, light layers, transitional pieces"
        elif season == "Fall":
            advice += "Earth tones, layering, boots, textured fabrics like knits and corduroys"
        else:
            advice += "Build versatile looks that work year-round with layering"
        
        return advice

def create_stylist():
    """Factory function to create stylist"""
    return GeminiStylist()