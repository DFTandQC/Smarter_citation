# kimi_extract_query.py
from openai import OpenAI
import json

class KimiExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract(self, paragraph: str, years_back: int = 10) -> dict:
        """
        Output strict JSON:
        {
          "topics": [...3-5],
          "keywords": [...5-10],
          "entities": [...0-6],
          "year_from": 2016,
          "year_to": 2025
        }
        """
        system_prompt = (
            "You are a scientific research assistant specializing in atmospheric aerosol science, "
            "nucleation processes, and aircraft engine emissions. "
            "Extract search-relevant information from the given paragraph and return ONLY valid JSON. "
            "Do not include any explanatory text or markdown formatting."
        )
        user_prompt = f"""
Paragraph:
{paragraph}

Research context: Atmospheric new particle formation, aerosols, nucleation processes, aircraft engine particles, new particle formation around airports.

Extract and return JSON with these fields:
1. "topics": 3-5 main research areas/themes (e.g., "new particle formation", "nucleation mechanism", "aircraft emissions")
2. "keywords": 5-10 specific search terms, methods, substances, acronyms (e.g., "NPF", "organic", "ion-induced nucleation", "engine PM emissions")
3. "entities": 0-6 relevant organizations, research groups, airports, or specific studies (e.g., "CERN", "CLOUD", "Frankfurt Airport"). Use empty array [] if none.
4. "year_from": earliest relevant publication year (consider field maturity and research recency)
5. "year_to": latest year (typically {2025 - years_back} to {2025})

Return ONLY valid JSON, no other text.
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        try:
            return json.loads(content)
        except Exception:
            # If JSON parsing fails, return empty dict for upstream fallback handling
            return {}
