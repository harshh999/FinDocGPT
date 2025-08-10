"""
Consolidated financial analysis utilities
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import Dict, List, Tuple, Optional
import requests
import json
import os
from .cache_manager import cache

class FinancialAnalyzer:
    """Consolidated financial analysis with optimized algorithms"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Compiled regex patterns for better performance
        self.financial_patterns = {
            'revenue': [
                re.compile(r'(?:total\s+)?(?:net\s+)?revenue[s]?\s*(?:of|was|is|:)?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|thousand|m|b|k)?', re.IGNORECASE),
                re.compile(r'(?:total\s+)?sales\s*(?:of|was|is|:)?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|thousand|m|b|k)?', re.IGNORECASE)
            ],
            'profit': [
                re.compile(r'net\s+income\s*(?:of|was|is|:)?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|thousand|m|b|k)?', re.IGNORECASE),
                re.compile(r'(?:net\s+)?profit\s*(?:of|was|is|:)?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|thousand|m|b|k)?', re.IGNORECASE)
            ],
            'expenses': [
                re.compile(r'(?:total\s+)?(?:operating\s+)?expenses\s*(?:of|was|is|:)?\s*\$?([0-9,]+\.?[0-9]*)\s*(?:million|billion|thousand|m|b|k)?', re.IGNORECASE)
            ]
        }
        
        # Sentiment keywords for faster lookup
        self.positive_words = set(['growth', 'increase', 'profit', 'strong', 'excellent', 'outstanding', 'improved', 'expansion', 'success'])
        self.negative_words = set(['decline', 'loss', 'decrease', 'weak', 'poor', 'challenging', 'difficult', 'concern', 'risk'])
    
    def extract_financial_metrics(self, text: str) -> Dict[str, float]:
        """Optimized financial metrics extraction using compiled regex"""
        metrics = {}
        
        for metric_type, patterns in self.financial_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    try:
                        value = self._parse_financial_value(match.group(1), match.group(0))
                        metrics[metric_type] = value
                        break
                    except (ValueError, IndexError):
                        continue
            if metric_type in metrics:
                continue
        
        return metrics
    
    def _parse_financial_value(self, value_str: str, context: str) -> float:
        """Parse financial value with unit conversion"""
        base_value = float(value_str.replace(',', ''))
        context_lower = context.lower()
        
        if 'billion' in context_lower or ' b' in context_lower:
            return base_value * 1e9
        elif 'million' in context_lower or ' m' in context_lower:
            return base_value * 1e6
        elif 'thousand' in context_lower or ' k' in context_lower:
            return base_value * 1e3
        
        return base_value
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Optimized sentiment analysis"""
        # Use TextBlob for baseline
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Fast keyword-based adjustment
        words = set(text.lower().split())
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        
        # Adjust polarity based on financial context
        if positive_count > negative_count:
            polarity = max(polarity, 0.1)
        elif negative_count > positive_count:
            polarity = min(polarity, -0.1)
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return sentiment, polarity
    
    def ai_powered_qa(self, text: str, question: str) -> str:
        """AI-powered Q&A with caching"""
        cache_params = {'text_hash': hash(text[:1000]), 'question': question}
        cached_answer = cache.get('qa_response', cache_params, 60)
        
        if cached_answer is not None:
            return cached_answer
        
        if self.api_key and self.api_key != "demo":
            answer = self._get_ai_answer(text, question)
        else:
            answer = self._get_fallback_answer(text, question)
        
        cache.set('qa_response', cache_params, answer)
        return answer
    
    def _get_ai_answer(self, text: str, question: str) -> str:
        """Get AI answer using Gemini API"""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            
            prompt = f"""You are a professional financial analyst. Analyze this financial document and provide a comprehensive, detailed answer to the question.

DOCUMENT CONTENT:
{text[:8000]}

QUESTION: {question}

Please provide a thorough, detailed response that:
1. Directly addresses the question with specific information from the document
2. Provides relevant context and background information
3. Includes specific numbers, dates, percentages, or financial figures when available
4. Explains the significance or implications of the findings
5. Offers additional insights or related information from the document
6. Uses professional financial terminology appropriately
7. Structures the response with clear sections or bullet points when helpful

Make your response comprehensive (200-400 words) while remaining accurate and based solely on the document content. If the document doesn't contain sufficient information to answer the question, explain what information is available and suggest related topics the document does cover."""

            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': self.api_key
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts and 'text' in parts[0]:
                            return parts[0]['text'].strip()
            
            return "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            return self._get_fallback_answer(text, question)
    
    def _get_fallback_answer(self, text: str, question: str) -> str:
        """Enhanced fallback Q&A with comprehensive responses"""
        question_lower = question.lower()
        question_words = [word.strip('?.,!()[]') for word in question_lower.split() 
                         if len(word) > 3 and word not in {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'the', 'and', 'or', 'but'}]
        
        # Split text into sentences and paragraphs
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 50]
        
        if not sentences:
            return "No content available to answer the question."
        
        # Enhanced sentence scoring with context
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Count keyword matches
            matches = sum(1 for word in question_words if word in sentence_lower)
            if matches == 0:
                continue
            
            # Base score from matches
            score = matches * 4
            
            # Boost for financial terms and numbers
            if re.search(r'\d+', sentence):
                score += 3
            if any(term in sentence_lower for term in ['million', 'billion', '%', 'quarter', 'year', 'revenue', 'profit', 'growth']):
                score += 2
            
            # Boost for important indicators
            if any(indicator in sentence_lower for indicator in ['important', 'key', 'main', 'significant', 'major']):
                score += 2
            
            # Position bonus for earlier sentences
            if i < len(sentences) * 0.3:  # First 30% of document
                score += 1
            
            sentence_scores.append((sentence, score, i))
        
        if not sentence_scores:
            return f"I couldn't find specific information to answer '{question}' in the document. The document appears to cover other topics that might be of interest."
        
        # Sort by score and get top matches
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:5]  # Get top 5 matches
        
        # Build comprehensive answer
        answer_parts = []
        
        # Main answer
        primary_sentence = top_sentences[0][0]
        answer_parts.append(f"**Primary Answer:**\n{primary_sentence}")
        
        # Supporting information
        if len(top_sentences) > 1:
            supporting_sentences = []
            for sentence, score, pos in top_sentences[1:3]:  # Next 2 best matches
                if score >= top_sentences[0][1] * 0.5:  # At least 50% as relevant
                    supporting_sentences.append(sentence)
            
            if supporting_sentences:
                answer_parts.append(f"\n\n**Supporting Information:**")
                for i, support in enumerate(supporting_sentences, 1):
                    answer_parts.append(f"\n{i}. {support}")
        
        # Extract and highlight key data
        all_relevant_text = ' '.join([item[0] for item in top_sentences[:3]])
        
        # Extract numbers and financial data
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|\s*(?:million|billion|thousand|M|B|K))?\b', all_relevant_text)
        dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|Q[1-4]\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', all_relevant_text)
        
        if numbers:
            unique_numbers = list(set(numbers))[:5]
            answer_parts.append(f"\n\n**Key Figures:** {', '.join(unique_numbers)}")
        
        if dates:
            unique_dates = list(set(dates))[:3]
            answer_parts.append(f"\n\n**Relevant Dates:** {', '.join(unique_dates)}")
        
        # Add context about document coverage
        if len(sentence_scores) > 3:
            answer_parts.append(f"\n\n**Additional Context:** The document contains {len(sentence_scores)} relevant sections related to your question, providing comprehensive coverage of this topic.")
        
        # Suggest related topics if available
        related_topics = []
        for word in question_words:
            for sentence, _, _ in sentence_scores[3:6]:  # Check next few sentences
                if word in sentence.lower() and sentence not in [item[0] for item in top_sentences[:3]]:
                    related_topics.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
                    break
        
        if related_topics:
            answer_parts.append(f"\n\n**Related Information:** {related_topics[0]}")
        
        return ''.join(answer_parts)
    
    def generate_document_summary(self, content: str, metadata: Dict) -> Dict:
        """Generate optimized document summary"""
        cache_params = {'content_hash': hash(content[:2000]), 'file_type': metadata.get('file_type', 'unknown')}
        cached_summary = cache.get('document_summary', cache_params, 120)
        
        if cached_summary is not None:
            return cached_summary
        
        # Extract metrics and sentiment
        metrics = self.extract_financial_metrics(content)
        sentiment, polarity = self.analyze_sentiment(content)
        
        # Extract key topics efficiently
        key_topics = self._extract_key_topics_fast(content)
        
        # Extract main points
        main_points = self._extract_main_points_fast(content)
        
        # Extract entities
        entities = self._extract_entities_fast(content)
        
        summary = {
            'overview': self._generate_overview(content, metadata),
            'key_topics': key_topics,
            'financial_highlights': metrics,
            'entities': entities,
            'main_points': main_points,
            'sentiment': {'label': sentiment, 'polarity': polarity}
        }
        
        cache.set('document_summary', cache_params, summary)
        return summary
    
    def _extract_key_topics_fast(self, content: str) -> Dict[str, int]:
        """Fast topic extraction using set operations"""
        content_words = set(content.lower().split())
        
        topic_keywords = {
            'Financial Performance': {'revenue', 'sales', 'profit', 'earnings', 'income', 'ebitda', 'margin'},
            'Growth': {'growth', 'increase', 'expansion', 'grew', 'rising'},
            'Risks': {'risk', 'challenge', 'concern', 'uncertainty', 'threat'},
            'Strategy': {'strategy', 'plan', 'initiative', 'goal', 'objective'},
            'Market': {'market', 'competition', 'industry', 'sector', 'share'},
            'Operations': {'operations', 'production', 'efficiency', 'cost', 'manufacturing'}
        }
        
        topics_found = {}
        for topic, keywords in topic_keywords.items():
            count = len(content_words & keywords)
            if count > 0:
                topics_found[topic] = count
        
        return dict(sorted(topics_found.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_main_points_fast(self, content: str) -> List[str]:
        """Fast main points extraction"""
        sentences = [s.strip() for s in content.split('.') if 50 < len(s.strip()) < 300]
        
        if not sentences:
            return []
        
        # Score sentences based on financial keywords
        financial_keywords = {'revenue', 'profit', 'growth', 'earnings', 'performance', 'results'}
        
        scored_sentences = []
        for i, sentence in enumerate(sentences[:20]):  # Limit to first 20 sentences
            sentence_words = set(sentence.lower().split())
            keyword_count = len(sentence_words & financial_keywords)
            
            if keyword_count > 0:
                # Position bonus (earlier sentences get higher scores)
                position_bonus = max(0, 10 - i)
                score = keyword_count * 2 + position_bonus
                scored_sentences.append((sentence, score))
        
        # Return top 5 sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in scored_sentences[:5]]
    
    def _extract_entities_fast(self, content: str) -> Dict[str, List[str]]:
        """Fast entity extraction using regex"""
        entities = {'Companies': [], 'Dates': []}
        
        # Company patterns
        company_matches = re.findall(r'\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|Company|Ltd|LLC)\b', content)
        entities['Companies'] = list(set(company_matches[:5]))
        
        # Date patterns
        date_matches = re.findall(r'\b(?:Q[1-4]\s+\d{4}|\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', content)
        entities['Dates'] = list(set(date_matches[:5]))
        
        return entities
    
    def _generate_overview(self, content: str, metadata: Dict) -> str:
        """Generate document overview"""
        file_type = metadata.get('file_type', 'document')
        word_count = len(content.split())
        
        # Count financial terms
        financial_terms = {'revenue', 'profit', 'earnings', 'growth', 'loss', 'expenses'}
        content_words = set(content.lower().split())
        financial_count = len(content_words & financial_terms)
        
        # Determine document purpose
        if financial_count > 5:
            purpose = "financial report or analysis"
        elif file_type == 'pptx':
            purpose = "business presentation"
        elif 'strategy' in content.lower():
            purpose = "strategic document"
        else:
            purpose = "business document"
        
        return f"This {file_type.upper()} file appears to be a {purpose} containing approximately {word_count:,} words with {financial_count} financial term references."

# Global analyzer instance
financial_analyzer = FinancialAnalyzer()