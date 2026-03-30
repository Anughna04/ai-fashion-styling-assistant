"""
Groq AI Stylist - Generates personalized fashion recommendations using Groq
"""

from groq import Groq
from config import GROQ_API_KEY

class GroqStylist:
    """AI Fashion Stylist using Groq (Fast and Reliable!)"""
    
    def __init__(self):
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
                self.available = True
                print("✅ Groq AI Stylist initialized (Lightning Fast!)")
            except Exception as e:
                self.available = False
                print(f"⚠️ Groq initialization failed: {e}")
        else:
            self.available = False
            print("⚠️ Groq API key not found - using fallback stylist")
            print("   Get your free key at: https://console.groq.com/keys")
    
    def generate_detailed_advice(self, items, user_query, occasion, season, image_description=None):
        """Generate comprehensive styling advice using Groq"""
        
        if not self.available:
            return self._fallback_advice(items, occasion, season, user_query)
        
        # Prepare item details
        items_detail = []
        for idx, item in enumerate(items[:6], 1):
            items_detail.append(
                f"{idx}. {item.get('name', 'Fashion Item')} - "
                f"Category: {item.get('category', 'N/A')}, "
                f"Style: {item.get('style', 'N/A')}, "
                f"Color: {item.get('color', 'N/A')}, "
                f"Fabric: {item.get('fabric', 'N/A')}, "
                f"For: {', '.join(item.get('occasions', []))}, "
                f"Seasons: {', '.join(item.get('seasons', []))}"
            )
        
        items_text = "\n".join(items_detail)
        
        # Create comprehensive prompt
        system_prompt = """You are an elite fashion stylist with 20+ years of experience working with celebrities and high-profile clients. You have expertise in:
- Color theory and seasonal palettes
- Body shape and proportion analysis
- Current fashion trends and timeless classics
- Luxury and affordable fashion brands
- Accessorizing and completing looks
- Fabric care and garment maintenance

Your advice is specific, actionable, and makes clients feel confident and stylish."""

        user_prompt = f"""A client needs your expert styling advice:

CLIENT REQUEST: "{user_query}"
OCCASION: {occasion or 'Versatile/Multiple occasions'}
SEASON: {season or 'All seasons'}
{f'IMAGE CONTEXT: {image_description}' if image_description else 'Text-based search'}

AVAILABLE ITEMS IN THEIR WARDROBE:
{items_text}

Provide a comprehensive styling guide with these sections:

## 🎨 OUTFIT COMBINATIONS (Be Specific!)

Create 2-3 complete outfits using the items above. For EACH outfit:
1. List exact item numbers to combine (e.g., "Combine Item 1 + Item 3")
2. Explain the styling reasoning (why these pieces work together)
3. Describe the overall vibe/aesthetic
4. Mention any tucking, layering, or styling techniques

## 💎 COLOR MASTERY

- Analyze the color palette of recommended items
- Explain color harmony principles (complementary, monochromatic, analogous)
- Suggest 2-3 accent colors to add via accessories
- Give undertone advice (warm/cool/neutral skin tones)
- Warn about color combinations to avoid

## ✨ JEWELRY & ACCESSORIES (Exact Details!)

For {occasion or 'this look'}:
- **Necklace**: Specific type (e.g., "16-inch gold pendant with geometric design"), when to wear, when to skip
- **Earrings**: Size and style (e.g., "1-inch silver hoops" or "statement 3-inch drops"), matching rules
- **Bracelets**: How many, style, stacking tips
- **Rings**: Which fingers, how many, statement vs delicate
- **Watch**: Leather/metal band, formal vs casual, when to wear
- **Bags**: Specific size and type (e.g., "medium crossbody 10x8 inches" or "small clutch")
- **Belts**: Width (1"/2"/3"), color, cinching techniques
- **Other**: Scarves, hats, sunglasses if relevant

## 👠 FOOTWEAR STRATEGY

Provide 3 shoe options with exact details:
1. **Option 1**: Shoe type, heel height, color, when to wear
2. **Option 2**: Alternative style, reasoning
3. **Option 3**: Comfortable/casual option

Include:
- Color coordination rules for shoes
- Sock/hosiery recommendations
- Comfort tips for high heels

## 🧵 FABRIC & TEXTURE INSIGHTS

- Which fabric combinations work (and why)
- Texture mixing tips (smooth + rough, matte + shiny)
- Seasonal appropriateness
- Layering for temperature control
- Quick care tips for each fabric type

## 📐 FIT & PROPORTION SECRETS

- Body-type specific advice
- Balance oversized and fitted pieces
- Tucking strategies (full/French/half/out)
- Where to cinch with belts
- Ideal hemlines and lengths
- Creating vertical lines for height

## 💄 COMPLETE THE LOOK

- Hair styling (up/down/textured/sleek)
- Makeup vibe (natural/glam/bold lip/smokey eye)
- Nail color and finish recommendations
- Fragrance notes that match the aesthetic
- Grooming details (neat/relaxed)

## 📈 TREND vs TIMELESS

- Current 2024-2025 trends incorporated
- Classic elements that never date
- How to trend-proof your investment pieces
- Which items will still look good in 5 years

## 🛍️ SHOPPING GUIDANCE

- Specific brands to check (luxury and affordable)
- Which pieces to invest in (spend more)
- Where to save (fast fashion options)
- Similar items to add to wardrobe
- How to rewear these pieces differently

## ⚠️ STYLE MISTAKES TO AVOID

- Common errors for {occasion or 'this occasion'}
- Color clashes to watch out for
- Over-accessorizing red flags
- Fit issues (too tight/loose/short/long)
- Fabric faux pas

Keep your tone warm, encouraging, and conversational. Use "you" to address the client directly. Give measurements, brand examples, and specific product types. Make them excited to try these looks!"""

        try:
            # Use Groq's fastest model
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Fast and high quality
                temperature=0.7,
                max_tokens=4000,
                top_p=0.9,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return self._fallback_advice(items, occasion, season, user_query)
    
    def _fallback_advice(self, items, occasion, season, query=""):
        """Enhanced fallback advice when API is unavailable"""
        
        # Categorize items
        tops = [i for i in items if i.get('category') == 'Top']
        bottoms = [i for i in items if i.get('category') == 'Bottom']
        dresses = [i for i in items if i.get('category') == 'Dress']
        outerwear = [i for i in items if i.get('category') == 'Outerwear']
        shoes = [i for i in items if i.get('category') == 'Footwear']
        jewelry = [i for i in items if i.get('category') == 'Jewelry']
        accessories = [i for i in items if i.get('category') == 'Accessories']
        
        advice = f"""## 🎨 Personalized Styling Guide

**⚠️ Using Basic Recommendations** - Add your Groq API key for AI-powered personalized advice!
Get free API key at: https://console.groq.com/keys

**Search Query:** "{query[:150]}"
**Occasion:** {occasion or 'Not specified'}
**Season:** {season or 'All seasons'}
**Items Found:** {len(items)} matching pieces

---

### 🌟 Complete Outfit Combinations

"""
        
        # Create outfit 1
        if tops and bottoms:
            advice += f"""**OUTFIT 1 - Classic & Polished:**
- **Top:** {tops[0]['name']} ({tops[0]['color']} {tops[0]['fabric']})
- **Bottom:** {bottoms[0]['name']} ({bottoms[0]['color']} {bottoms[0]['fabric']})
"""
            if outerwear:
                advice += f"- **Layer:** {outerwear[0]['name']} for added sophistication\n"
            if shoes:
                advice += f"- **Shoes:** {shoes[0]['name']}\n"
            
            advice += f"\n**Why it works:** The {tops[0]['color']} and {bottoms[0]['color']} create a {tops[0]['style'].lower()} vibe perfect for {', '.join(tops[0].get('occasions', ['any occasion']))}.\n\n"
        
        # Create outfit 2
        if dresses:
            advice += f"""**OUTFIT 2 - Statement Piece:**
- **Dress:** {dresses[0]['name']} ({dresses[0]['color']} {dresses[0]['fabric']})
"""
            if shoes:
                advice += f"- **Shoes:** {shoes[0]['name']}\n"
            if jewelry:
                advice += f"- **Jewelry:** {jewelry[0]['name']}\n"
            
            advice += f"\n**Why it works:** This {dresses[0]['color']} {dresses[0]['fabric']} dress is a complete look that shines for {', '.join(dresses[0].get('occasions', ['special events']))}.\n\n"
        
        # Create outfit 3
        if len(tops) > 1 and len(bottoms) > 1:
            advice += f"""**OUTFIT 3 - Mix & Match:**
- **Top:** {tops[1]['name']}
- **Bottom:** {bottoms[1]['name']}
"""
            if len(shoes) > 1:
                advice += f"- **Shoes:** {shoes[1]['name']}\n"
            
            advice += f"\n**Styling tip:** Try a French tuck with the top for a relaxed yet put-together look.\n\n"
        
        # Color coordination
        colors = list(set([i['color'] for i in items[:5]]))
        advice += f"""---

### 🎨 Color Coordination

**Your Color Palette:** {', '.join(colors)}

"""
        
        if 'Black' in colors or 'Navy' in colors or 'Grey' in colors:
            advice += "✓ **Neutral Foundation:** Black, navy, and grey are your versatile bases - build around these\n"
        
        if any(c in colors for c in ['Red', 'Coral', 'Burgundy', 'Orange', 'Yellow']):
            advice += "✓ **Warm Tones Present:** Pair with gold jewelry, camel, brown, or cream accessories\n"
        
        if any(c in colors for c in ['Blue', 'Purple', 'Green']):
            advice += "✓ **Cool Tones Present:** Silver jewelry, grey, and white will complement beautifully\n"
        
        advice += f"""
**Color Theory Tips:**
- 60-30-10 Rule: 60% dominant color, 30% secondary, 10% accent
- Complementary colors create visual interest (e.g., blue + orange)
- Monochrome looks (all one color family) are ultra-chic
- White/cream brightens any outfit
- Black adds instant sophistication

"""
        
        # Occasion-specific jewelry
        advice += f"""---

### ✨ Jewelry & Accessories for {occasion or 'Your Look'}

"""
        
        if occasion == 'Formal' or occasion == 'Party':
            advice += """**Formal/Party Styling:**

**Jewelry:**
- **Necklace:** Statement piece (18-20" length) OR skip if wearing dramatic earrings
- **Earrings:** Chandelier or drop earrings (2-3" length) if no necklace
- **Bracelets:** Delicate bangles or single cuff - don't overdo it
- **Rings:** 1-2 cocktail rings on different fingers

**Metal Choice:** Gold for warm-toned outfits, silver for cool tones, rose gold for romantic looks

**Bags:** Small evening clutch (6x4") in metallic, satin, or matching color

**Other:**
- Add a silk scarf draped over shoulders for elegance
- Consider a statement belt to cinch waist on dresses
"""
        
        elif occasion == 'Business':
            advice += """**Business Professional:**

**Jewelry:**
- **Necklace:** Simple 16-18" pendant or skip entirely
- **Earrings:** Small studs or huggie hoops (under 1cm)
- **Watch:** Leather or metal band, classic face - always wear one
- **Rings:** Maximum 2, keep understated

**Rule:** Can see jewelry from 3 feet away = too much for business

**Bags:** Structured tote (13x10") or satchel in black, navy, brown leather

**Other:**
- Belt should match shoes
- No jangly bracelets that make noise
- Keep it polished and minimal
"""
        
        elif occasion == 'Casual':
            advice += """**Casual Everyday:**

**Jewelry:**
- **Necklace:** Layered delicate chains (2-3) or simple pendant
- **Earrings:** Studs, small hoops, or fun statement pieces - express yourself!
- **Bracelets:** Stack 2-4 thin bracelets or wear single cuff
- **Watch:** Sporty or casual leather band

**Mix metals freely!** Gold + silver together is totally on-trend

**Bags:** Crossbody (9x7") for hands-free convenience or tote (14x12") for carrying everything

**Other:**
- Baseball cap or beanie for casual cool
- Sunglasses add instant style
- Canvas sneakers or ankle boots complete the look
"""
        
        else:
            advice += """**Versatile Accessorizing:**

**The Formula:**
1. Choose ONE statement piece (bold necklace OR dramatic earrings OR eye-catching bag)
2. Keep everything else subtle and complementary
3. Match metal tones OR intentionally mix for modern look
4. Consider the neckline: V-neck = pendant, crew neck = no necklace or long chain

**Jewelry Stacking:**
- Necklaces: Vary lengths (16", 18", 24") for layering
- Bracelets: Mix thin + thick, metal + leather
- Rings: Spread across both hands, vary sizes

**Bag Size Guide:**
- Clutch (6x4"): Evening/formal
- Crossbody (9x7"): Casual daytime  
- Tote (14x12"): Work/errands
- Satchel (12x10"): Professional/structured
"""
        
        # Footwear
        advice += f"""---

### 👠 Footwear Selection Guide

"""
        
        if shoes:
            advice += f"""**Your Matched Shoes:**
- {shoes[0]['name']} - {shoes[0].get('styling_tip', 'Perfect choice')}
"""
            if len(shoes) > 1:
                advice += f"- {shoes[1]['name']} - Great alternative\n"
        
        advice += f"""
**By Occasion & Heel Height:**

| Occasion | Shoe Type | Heel Height | Colors |
|----------|-----------|-------------|---------|
| **Casual** | Sneakers, loafers, flat sandals | Flat to 1" | White, tan, navy |
| **Business** | Closed-toe pumps, loafers, oxfords | 2-3" | Black, navy, nude |
| **Formal** | Classic pumps, strappy heels | 3-4" | Black, nude, metallic |
| **Party** | Statement heels, velvet, embellished | 3-5" | Metallic, jewel tones |
| **Date** | Block heels, heeled booties | 2.5-3.5" | Nude, black, burgundy |

**Color Matching Rules:**
- **Nude/Beige:** Lengthens legs, works with everything
- **Black:** Classic, slimming, ultra-versatile  
- **Metallic Silver:** Pairs with cool colors (blue, purple, grey)
- **Metallic Gold:** Complements warm colors (red, orange, brown)
- **Match your bag:** Traditional rule for polished look
- **Contrast your bag:** Modern approach for visual interest

**Comfort Hacks:**
- Add gel inserts for high heels
- Break in shoes at home before wearing out
- Bring foldable flats in purse for heel relief
- Choose block heels over stilettos for all-day comfort
"""
        
        # Seasonal guidance
        advice += f"""---

### 🌸 Seasonal Styling for {season or 'Year-Round'}

"""
        
        seasonal_guides = {
            'Spring': """**Spring Fashion Guide:**

**Color Palette:**
- Pastels: Blush pink, mint green, lavender, powder blue
- Fresh whites and creams
- Soft yellows and peach tones

**Fabrics to Wear:**
- Cotton (breathable and fresh)
- Linen (gets better with wrinkles!)
- Light knits and chambray
- Silk and chiffon for dressier looks

**Layering Strategy:**
- Light cardigan or denim jacket
- Trench coat for rain
- Silk scarf for variable temps

**Footwear:**
- Loafers and oxford shoes
- White sneakers (always chic)
- Ankle boots (transitional)
- Open-toe sandals as it warms

**Spring Accessories:**
- Straw bags and woven textures
- Delicate floral silk scarves
- Pastel sunglasses
- Garden party hats

**Avoid:** Heavy fabrics, dark winter colors, knee-high boots (too warm)""",

            'Summer': """**Summer Fashion Guide:**

**Color Palette:**
- Crisp whites and off-whites
- Bright colors: Coral, turquoise, yellow, hot pink
- Tropical prints and nautical stripes
- Soft neutrals: Sand, cream, tan

**Fabrics to Wear:**
- Linen (most breathable!)
- Cotton and cotton blends
- Chambray and seersucker
- Light silk and rayon

**Key Pieces:**
- Sleeveless tops and sundresses
- Shorts and cropped pants
- Breathable wide-leg trousers
- Light kimonos for coverage

**Footwear:**
- Flat sandals and slides
- Espadrille wedges (comfortable + chic)
- Canvas sneakers
- Strappy flat sandals

**Summer Accessories:**
- Wide-brim sun hat (protection + style)
- Woven or raffia bags
- Oversized sunglasses
- Minimal jewelry (less is more in heat)

**Avoid:** Heavy knits, boots, dark colors in direct sun, tight synthetic fabrics""",

            'Fall': """**Fall Fashion Guide:**

**Color Palette:**
- Earth tones: Rust, olive, mustard, terracotta
- Rich jewel tones: Burgundy, emerald, sapphire
- Warm neutrals: Camel, chocolate brown, cream
- Classic autumn: Burnt orange, deep red

**Fabrics to Wear:**
- Wool and wool blends
- Corduroy (trending!)
- Suede and leather
- Chunky knits and cashmere
- Denim (perfect weight)

**Layering Mastery:**
- Start with fitted base layer
- Add sweater or cardigan
- Top with blazer or leather jacket
- Finish with scarf

**Footwear:**
- Ankle boots (the MVP of fall)
- Loafers and oxfords
- Knee-high boots
- Chunky sneakers

**Fall Accessories:**
- Wool or cashmere scarves
- Felt hats and beanies
- Leather bags and crossbodies
- Tights and socks become fashion

**Avoid:** Summer sandals, lightweight fabrics, white after Labor Day (traditional, not mandatory)""",

            'Winter': """**Winter Fashion Guide:**

**Color Palette:**
- Deep jewel tones: Sapphire, emerald, ruby red
- Classic darks: Black, charcoal, navy
- Winter whites and creams
- Burgundy and forest green

**Fabrics to Wear:**
- Wool and cashmere (warmth + luxury)
- Heavy knits and cable sweaters
- Velvet (festive and warm)
- Quilted and down (practical)
- Fleece-lined (hidden comfort)

**Layering Strategy:**
- Thermal base layer
- Turtleneck or thick knit
- Wool blazer or cardigan  
- Coat (wool, puffer, or parka)
- Scarf, gloves, hat

**Footwear:**
- Knee-high boots
- Combat boots (waterproof)
- Ankle boots with socks
- Insulated sneakers

**Winter Accessories:**
- Wool or cashmere scarves (oversized)
- Leather gloves (lined)
- Knit beanies or berets
- Structured leather bags

**Avoid:** Exposed ankles, summer fabrics, open-toe shoes, skipping the coat for style"""
        }
        
        advice += seasonal_guides.get(season, """**Year-Round Capsule Wardrobe:**

Build these essentials for mixing and matching across all seasons:

**Tops (6):**
- White button-up shirt
- Black turtleneck
- Striped long-sleeve tee
- Silk blouse (neutral color)
- Cashmere or quality knit sweater
- Denim jacket

**Bottoms (5):**
- Dark wash jeans (well-fitted)
- Black pants/trousers
- Neutral midi skirt
- Tailored shorts
- Wide-leg trousers

**Dresses (3):**
- Little black dress
- Midi shirt dress
- Casual day dress

**Outerwear (3):**
- Trench coat (transitional)
- Wool coat (winter)
- Leather or denim jacket

**Shoes (5):**
- White sneakers
- Black ankle boots
- Nude pumps
- Casual flats
- Statement heels

**Quality Over Quantity:** Invest in items you'll wear 100+ times""")
        
        # Pro tips
        advice += f"""

---

### 💡 Professional Styling Tips

**The 3-Layer Rule:**
1. Base layer (fitted)
2. Middle layer (adds interest)
3. Outer layer (pulls it together)

**Proportion Secrets:**
- Tight on top = loose on bottom (and vice versa)
- Show skin somewhere: ankles, wrists, collarbone, or décolletage
- High-waisted bottoms + cropped top = leg lengthening
- Vertical lines (stripes, long necklaces) add height

**Tucking Techniques:**
- **Full Tuck:** Formal, polished, shows belt/waistline
- **French Tuck (Half):** Casual-chic, front tucked only
- **Side Tuck:** Asymmetric, trendy, flattering
- **No Tuck:** Oversized, relaxed, or intentionally long tops

**Fit Fixes:**
- Tailor everything: Even $30 pants look $300 when tailored
- Shoulders should fit perfectly (hardest to alter)
- Sleeves should hit wrist bone or be intentionally cropped
- Pants should not puddle on shoes (hem them!)

**The Mirror Test:**
- Front view: Balanced proportions?
- Side view: Smooth lines, no pulling?
- Back view: Neat and intentional? (people see this most!)
- Sit down test: Still comfortable and modest?

---

### 🎯 Style Dos and Don'ts

**DO:**
✓ Invest in quality basics that fit perfectly
✓ Mix high and low price points freely
✓ Wear what makes YOU feel confident
✓ Follow the 3-color rule (max 3 main colors per outfit)
✓ Check weather and venue before choosing outfit
✓ Steam or iron clothes (wrinkles ruin expensive looks)
✓ Build outfits around one statement piece

**DON'T:**
✗ Match your bag and shoes exactly (too matchy-matchy, outdated rule)
✗ Wear more than one statement piece simultaneously
✗ Ignore proportions (all tight or all loose both unflattering)
✗ Forget to check the back view in mirror
✗ Wear clothes too small (well-fitted ≠ tight)
✗ Over-accessorize (less is almost always more)
✗ Follow trends that don't suit your body or style

---

### 🛍️ Shopping Strategy

**Investment Pieces (Spend More):**
- Classic leather handbag
- Wool coat
- Well-fitted blazer
- Quality leather shoes/boots
- Tailored trousers
- Cashmere sweater

**Save Money On (Fast Fashion OK):**
- Trendy pieces you'll wear one season
- Basic tees and tanks
- Costume jewelry
- Seasonal accessories
- Athleisure for gym only

**Brand Suggestions:**
- **Luxury**: Reformation, Everlane, COS, & Other Stories
- **Mid-Range**: Zara, Mango, Nordstrom, J.Crew
- **Budget**: H&M, Uniqlo, Target, ASOS
- **Shoes**: Nisolo, Everlane, Sam Edelman, Steve Madden
- **Bags**: Madewell, Portland Leather, Fossil

---

## 🚀 Ready to Upgrade?

For AI-powered personalized recommendations analyzing YOUR specific items, colors, and body type:
1. Get free Groq API key: https://console.groq.com/keys
2. Add to .env file: `GROQ_API_KEY=your_key_here`
3. Get detailed outfit combinations, color analysis, and trend insights!

---

*Happy styling! Remember: Fashion is about expressing YOUR unique personality. Wear what makes you feel amazing!* ✨
"""
        
        return advice

def create_stylist():
    """Factory function to create Groq stylist"""
    return GroqStylist()