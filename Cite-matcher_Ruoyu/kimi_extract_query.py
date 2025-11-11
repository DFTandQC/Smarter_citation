# kimi_extract_query.py
from openai import OpenAI
import json

class KimiExtractor:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract(self, paragraph: str, years_back: int = 10) -> dict:
        """
        输出严格 JSON：
        {
          "topics": [...3-5],
          "keywords": [...5-10],
          "entities": [...0-6],
          "year_from": 2016,
          "year_to": 2025
        }
        """
        system_prompt = (
            "你是一个科研助理，任务是从论文段落中抽取用于学术检索的核心信息。"
            "只返回严格 JSON，键包括：topics（3-5个主题短语），"
            "keywords（5-10个高价值检索词），entities（0-6个专名），"
            "year_from，year_to。不要输出任何除 JSON 以外的内容。"
        )
        user_prompt = f"""
段落内容：
{paragraph}

请抽取：
1. topics: 3-5 个主题短语；
2. keywords: 5-10 个高价值检索词（包括英文术语、化学式、方法名）；
3. entities: 0-6 个重要实体（作者、机构、化学物质、模型名等）；
4. year_from, year_to: 估计适合的文献时间窗（默认近 {years_back} 年）。
严格输出 JSON。
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
            # 若未能解析为 JSON，则返回空字典并让上游处理兜底
            return {}
