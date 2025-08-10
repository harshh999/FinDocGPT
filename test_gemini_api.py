#!/usr/bin/env python3
"""
Test script for Gemini API integration
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv('venv/.env')

class GeminiAPI:
    """Gemini API integration for document analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')  # Using the key from .env
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
    def generate_content(self, prompt):
        """Generate content using Gemini API"""
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            print(f"Making API request to: {self.base_url}")
            print(f"Using API key: {self.api_key[:10]}...")
            
            response = requests.post(self.base_url, headers=headers, json=data)
            print(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            print("Response received successfully!")
            
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print("No candidates in response")
                print(f"Full response: {result}")
                return "No response generated"
                
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return None

def test_gemini_api():
    """Test the Gemini API integration"""
    print("Testing Gemini API integration...")
    
    # Initialize API
    gemini = GeminiAPI()
    
    # Test simple prompt
    test_prompt = "Explain how AI works in a few words"
    print(f"\nTesting with prompt: {test_prompt}")
    
    result = gemini.generate_content(test_prompt)
    
    if result:
        print(f"\nSuccess! Response: {result}")
        return True
    else:
        print("\nFailed to get response from API")
        return False

def test_document_analysis():
    """Test document analysis functionality"""
    print("\n" + "="*50)
    print("Testing Document Analysis...")
    
    gemini = GeminiAPI()
    
    # Sample financial document content
    sample_content = """
    FINANCIAL REPORT Q3 2024
    
    Revenue: $2.5 million (up 15% from Q2)
    Net Income: $450,000 (up 8% from Q2)
    Operating Expenses: $1.8 million
    Cash Flow: $600,000 positive
    
    Key Highlights:
    - Strong growth in digital services segment
    - Successful product launch in European markets
    - Improved operational efficiency
    
    Risks:
    - Supply chain disruptions
    - Increased competition
    - Economic uncertainty
    """
    
    metadata = {
        'file_type': 'txt',
        'combined_words': len(sample_content.split()),
        'total_files': 1
    }
    
    prompt = f"""
    As a senior financial analyst, provide a comprehensive and detailed analysis of the following document. Each section should be thorough and precise, with 8-10 lines of detailed insights.

    Document Metadata:
    - File Type: {metadata.get('file_type', 'Unknown')}
    - Size: {metadata.get('combined_words', 0)} words
    - Files: {metadata.get('total_files', 1)}

    Document Content:
    {sample_content}

    Please provide a detailed analysis with the following sections:

    **1. EXECUTIVE SUMMARY (8-10 lines)**
    Provide a comprehensive overview of the document's purpose, scope, and primary focus. Detail the main business context, time period covered, and overall strategic direction. Explain the document's significance within the broader business landscape and its intended audience. Highlight the most critical information that stakeholders need to understand. Include any major announcements, strategic shifts, or significant developments mentioned. Assess the document's completeness and reliability as a source of business intelligence. Conclude with the overall business health and trajectory indicated by the content.

    **2. KEY FINANCIAL METRICS & PERFORMANCE (8-10 lines)**
    Extract and analyze all numerical data, financial figures, percentages, and quantitative metrics present in the document. Calculate growth rates, margins, and ratios where possible. Compare current performance against historical data or benchmarks mentioned. Identify trends in revenue, profitability, cash flow, and operational efficiency. Highlight any concerning financial indicators or exceptional performance areas. Assess the financial stability and liquidity position based on available data. Evaluate the quality of earnings and sustainability of financial performance. Provide context for each metric within industry standards and market conditions.

    **3. STRATEGIC THEMES & BUSINESS FOCUS (8-10 lines)**
    Identify the primary strategic initiatives, business priorities, and operational focus areas discussed in the document. Analyze the company's competitive positioning and market approach. Examine any mentions of digital transformation, innovation, or technological advancement. Assess the organization's approach to market expansion, customer acquisition, and product development. Evaluate the emphasis on operational excellence, cost management, and efficiency improvements. Identify any strategic partnerships, acquisitions, or investment priorities. Analyze the company's approach to sustainability, ESG factors, and corporate responsibility. Determine the alignment between stated strategy and operational execution.

    **4. MARKET SENTIMENT & OUTLOOK ANALYSIS (8-10 lines)**
    Conduct a thorough sentiment analysis of the language, tone, and messaging throughout the document. Assess the level of confidence, optimism, or caution expressed by management or authors. Evaluate the forward-looking statements and guidance provided for future performance. Analyze any mentions of market conditions, economic environment, and external factors. Determine the overall market positioning and competitive confidence expressed. Assess the transparency and candor of the communication style. Identify any defensive language, hedging, or uncertainty indicators. Evaluate the balance between achievements highlighted and challenges acknowledged. Provide a sentiment score (1-10) with detailed justification.

    **5. CRITICAL INSIGHTS & KEY TAKEAWAYS (8-10 lines)**
    Synthesize the most important insights that investors, stakeholders, or decision-makers should understand from this document. Identify any surprising revelations, unexpected developments, or significant changes from previous communications. Highlight the most compelling value propositions or competitive advantages presented. Assess the credibility and achievability of stated goals and projections. Identify any gaps in information or areas requiring further clarification. Evaluate the document's implications for different stakeholder groups (investors, employees, customers, partners). Determine the most actionable intelligence for strategic decision-making. Conclude with the three most critical points that define the organization's current position and future trajectory.

    Format your response with clear section headers and detailed, professional analysis for each area.
    """
    
    print("Analyzing sample financial document...")
    result = gemini.generate_content(prompt)
    
    if result:
        print(f"\nDocument Analysis Result:\n{result}")
        return True
    else:
        print("\nFailed to analyze document")
        return False

if __name__ == "__main__":
    print("Gemini API Test Suite")
    print("="*50)
    
    # Test basic API functionality
    basic_test = test_gemini_api()
    
    if basic_test:
        # Test document analysis
        doc_test = test_document_analysis()
        
        if doc_test:
            print("\n" + "="*50)
            print("✅ All tests passed! Gemini API integration is working correctly.")
        else:
            print("\n" + "="*50)
            print("❌ Document analysis test failed.")
    else:
        print("\n" + "="*50)
        print("❌ Basic API test failed. Check your API key and connection.")