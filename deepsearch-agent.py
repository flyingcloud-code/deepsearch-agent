# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import wikipedia # Changed from wikipediaapi
import arxiv
import json
import time
import random
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Set, Tuple
from dotenv import load_dotenv
from openai import AzureOpenAI

# Import API-based search libraries
from duckduckgo_search import DDGS
from googlesearch import search

# Load environment variables
load_dotenv()

# --- 配置 ---
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
DEFAULT_NUM_RESULTS = 3 # Reduce default for iterative search
DEFAULT_TIMEOUT = 10
FETCH_DELAY_RANGE = (0.5, 1.5)
MAX_ITERATIONS = 3 # Maximum research iterations
MAX_TOTAL_RESULTS = 20 # Limit total results accumulated across iterations
SLEEP_INTERVAL = random.uniform(2, 5)  # Sleep interval for API calls

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 辅助函数 ---
def _random_delay():
    time.sleep(random.uniform(*FETCH_DELAY_RANGE))

# --- API-Based Search Functions ---

def duckduckgo_search_api(query, max_results=5):
    """
    Perform a search using DuckDuckGo API with fallback mechanisms
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of search results
    """
    logging.info(f"Searching DuckDuckGo for: {query}")
    
    # Add request timeout for more stability
    timeout = 30  # seconds
    retry_count = 3
    
    for attempt in range(retry_count):
        try:
            with DDGS() as ddgs:
                # Try with longer timeout
                results = list(ddgs.text(query, max_results=max_results, timeout=timeout))
            
            if results:
                logging.info(f"Found {len(results)} results on attempt {attempt+1}")
                return results
            else:
                logging.warning(f"No results found on attempt {attempt+1}, retrying...")
        except Exception as e:
            logging.warning(f"Error on attempt {attempt+1}: {e}")
            # Implement backoff between retries
            if attempt < retry_count - 1:  # Don't sleep after the last attempt
                sleep_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    # If we exhaust all retries or get no results, try alternate approach: use our previous HTML scraper
    logging.warning("All API attempts failed, falling back to direct request method")
    return fallback_duckduckgo_search(query, max_results)

def fallback_duckduckgo_search(query, max_results=5):
    """Fallback to a more direct request if the DDGS API fails"""
    results = []
    try:
        # Use a direct request to DuckDuckGo's HTML interface
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        logging.info("Trying fallback direct HTML request to DuckDuckGo")
        response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        result_blocks = soup.find_all('div', class_='result')
        
        for i, block in enumerate(result_blocks):
            if i >= max_results:
                break
                
            link_tag = block.find('a', class_='result__a', href=True)
            snippet_tag = block.find('a', class_='result__snippet')
            
            if link_tag and snippet_tag:
                link = link_tag['href']
                # Handle redirect links
                if link.startswith("//duckduckgo.com/l/"):
                    try:
                        import urllib.parse
                        parsed_url = urllib.parse.urlparse(link)
                        qs = urllib.parse.parse_qs(parsed_url.query)
                        if 'uddg' in qs and qs['uddg']:
                            link = qs['uddg'][0]
                    except Exception:
                        pass
                        
                title = link_tag.get_text().strip()
                snippet = snippet_tag.get_text().strip().replace('\n', ' ')
                
                if link.startswith('http'):
                    results.append({
                        "title": title,
                        "href": link,
                        "body": snippet
                    })
        
        if results:
            logging.info(f"Fallback found {len(results)} results")
        else:
            logging.warning("Fallback found no results")
            
        return results
            
    except Exception as e:
        logging.error(f"Fallback search also failed: {e}")
        # Return empty results after all methods have failed
        return []

def google_search_api(query, max_results=5, lang="en", fetch_content=False, content_length=200):
    """
    Perform a search using Google Search
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        lang (str): Language for search results
        fetch_content (bool): Whether to fetch and extract content from the result URLs
        content_length (int): Maximum length of content to extract
        
    Returns:
        list: List of search results
    """
    logging.info(f"Searching Google for: {query}")
    
    try:
        # Using `search` from googlesearch with proper parameter names
        results = list(search(query, sleep_interval=SLEEP_INTERVAL, num_results=max_results,  lang=lang))
        
        formatted_results = []
        for i, result in enumerate(results):
            # Handle simple string URLs
            if isinstance(result, str):
                url = result
                title = f"Result {i+1}"
                snippet = "No description available"
            else:
                try:
                    url = result.url if hasattr(result, 'url') else result
                    title = result.title if hasattr(result, 'title') else f"Result {i+1}"
                    snippet = result.description if hasattr(result, 'description') else "No description available"
                except AttributeError:
                    url = str(result)
                    title = f"Result {i+1}"
                    snippet = "No description available"
            
            result_dict = {
                "title": title,
                "link": url,
                "snippet": snippet
            }
            
            if fetch_content:
                logging.info(f"Fetching content for result {i+1}/{len(results)}")
                content = fetch_webpage_content_api(url, content_length, i, len(results))
                if content:
                    result_dict["content"] = content
            
            formatted_results.append(result_dict)
        
        logging.info(f"Found {len(formatted_results)} results")
        return formatted_results
    except Exception as e:
        logging.error(f"Error searching Google: {e}")
        return []

def fetch_webpage_content_api(url, max_length=200, index=0, total=0):
    """
    Fetch and extract the main content from a webpage
    
    Args:
        url (str): URL to fetch
        max_length (int): Maximum length of content to return
        index (int): Index of the current result
        total (int): Total number of results
        
    Returns:
        str: Extracted content or None if failed
    """
    try:
        headers = {
            "User-Agent": DEFAULT_HEADERS['User-Agent']
        }
        response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            # Use BeautifulSoup to parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.extract()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean text (remove excess whitespace)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Add delay to avoid too frequent requests
            if index < total - 1:
                time.sleep(SLEEP_INTERVAL)
            
            return text[:max_length] + "..." if len(text) > max_length else text
        else:
            logging.error(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

# --- Azure OpenAI Integration ---
def init_azure_openai_client():
    """Initialize and return the Azure OpenAI client from environment variables"""
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment_name:
            logging.warning("AZURE_OPENAI_DEPLOYMENT_NAME not found in environment variables")
            return None
        return client
    except Exception as e:
        logging.error(f"Error initializing Azure OpenAI client: {e}")
        return None

# --- Placeholder LLM Functions ---
def llm_placeholder_plan(query: str, current_findings: str) -> List[str]:
    """
    Placeholder implementation for LLM planning function.
    Generates simple sub-queries based on the main query.
    """
    logging.info("Using placeholder planning function")
    
    # Create some basic sub-queries from the main query
    sub_queries = []
    
    # Add the original query
    sub_queries.append(query)
    
    # Add "definition" query
    sub_queries.append(f"definition of {query}")
    
    # Add "examples" query
    sub_queries.append(f"examples of {query}")
    
    # Add "latest research" query if appropriate
    if "research" in query.lower() or "study" in query.lower() or "science" in query.lower():
        sub_queries.append(f"latest research on {query}")
    
    logging.info(f"Generated {len(sub_queries)} placeholder sub-queries")
    return sub_queries

def llm_placeholder_synthesize(query: str, all_results: Dict[str, List[Dict[str, str]]], all_content: Dict[str, Optional[str]]) -> str:
    """
    Placeholder implementation for LLM synthesis function.
    Creates a simple structured report from search results.
    """
    logging.info("Using placeholder synthesis function")
    
    # Count total results
    total_results = sum(len(results) for results in all_results.values())
    
    # Create markdown report
    report = f"# Research Report: {query}\n\n"
    report += f"*This is an automatically generated research report on: {query}*\n\n"
    report += f"## Summary\n\n"
    report += f"This report compiles information from {total_results} search results across {len(all_results)} search engines.\n\n"
    
    # Add results by search engine
    report += f"## Search Results\n\n"
    for engine_name, results in all_results.items():
        if results:
            report += f"### {engine_name}\n\n"
            for i, result in enumerate(results):
                title = result.get('title', 'Untitled')
                link = result.get('link', result.get('href', '#'))
                snippet = result.get('snippet', result.get('body', 'No description available'))
                
                report += f"**{i+1}. [{title}]({link})**\n\n"
                report += f"{snippet}\n\n"
    
    # Add content extracts if available
    if all_content:
        report += f"## Content Extracts\n\n"
        for url, content in all_content.items():
            if content:
                # Get a short version of the URL for display
                display_url = url.replace('https://', '').replace('http://', '').split('/')[0]
                report += f"### Extract from {display_url}\n\n"
                # Limit content length for readability
                display_content = content[:1000] + "..." if len(content) > 1000 else content
                report += f"{display_content}\n\n"
    
    # Add conclusion
    report += f"## Conclusion\n\n"
    report += f"This report provides an overview of information related to '{query}'. "
    report += f"For more detailed information, please consult the original sources linked above.\n\n"
    report += f"*Report generated on: {time.strftime('%Y-%m-%d')}*"
    
    return report

def llm_placeholder_reflect(current_findings: str) -> Tuple[bool, List[str]]:
    """
    Placeholder implementation for LLM reflection function.
    Always suggests continuing research with some generic sub-queries.
    
    Returns:
        (needs_more_research: bool, new_queries: List[str])
    """
    logging.info("Using placeholder reflection function")
    
    # Count findings length as rough proxy for research completeness
    findings_length = len(current_findings)
    
    # Decide if more research is needed based on content length (arbitrary threshold)
    needs_more_research = findings_length < 5000
    
    # Generate new queries if more research is suggested
    new_queries = []
    if needs_more_research:
        # These are generic follow-up queries that might be useful in many research contexts
        new_queries = [
            "latest developments",
            "comparative analysis",
            "practical applications",
            "future directions"
        ]
    
    logging.info(f"Reflection result: needs_more={needs_more_research}, new_queries={len(new_queries)}")
    return needs_more_research, new_queries

# --- API-Based Search Functions ---
def duckduckgo_search_api(query, max_results=5):
    """
    Perform a search using DuckDuckGo API with fallback mechanisms
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of search results
    """
    logging.info(f"Searching DuckDuckGo for: {query}")
    
    # Add request timeout for more stability
    timeout = 30  # seconds
    retry_count = 3
    
    for attempt in range(retry_count):
        try:
            with DDGS() as ddgs:
                # Try with longer timeout
                results = list(ddgs.text(query, max_results=max_results, timelimit='y'))
            
            if results:
                logging.info(f"Found {len(results)} results on attempt {attempt+1}")
                return results
            else:
                logging.warning(f"No results found on attempt {attempt+1}, retrying...")
        except Exception as e:
            logging.warning(f"Error on attempt {attempt+1}: {e}")
            # Implement backoff between retries
            if attempt < retry_count - 1:  # Don't sleep after the last attempt
                sleep_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    # If we exhaust all retries or get no results, return empty list
    logging.warning("All API attempts failed, returning empty results")
    return []

# --- Real LLM Implementations using Azure OpenAI ---
def llm_plan(query: str, current_findings: str) -> List[str]:
    """
    Real implementation of LLM-based planning using Azure OpenAI.
    Takes the main query and current findings, returns a list of sub-queries.
    """
    client = init_azure_openai_client()
    if not client:
        logging.warning("Using placeholder planning function as OpenAI client initialization failed")
        return llm_placeholder_plan(query, current_findings)
    
    logging.info("Using real OpenAI model for planning")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    try:
        # Create a system message for planning
        system_message = """You are a research planning assistant. 
Your task is to analyze a research query and current findings, 
then generate 3-5 focused sub-queries that will help explore the topic thoroughly.
Provide these sub-queries as a JSON array of strings."""
        
        current_findings_summary = current_findings[:2000] if len(current_findings) > 2000 else current_findings
        
        user_message = f"""Main research query: "{query}"
Current findings summary: {current_findings_summary if current_findings else "No findings yet."}

Based on this information, please generate 3-5 focused sub-queries that will help explore different aspects of the main query.
Return ONLY a JSON array of strings, with no other text or explanation."""
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the JSON result to get the list of sub-queries
        try:
            # Try to parse the JSON result
            import json
            sub_queries = json.loads(result)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                logging.info(f"Generated {len(sub_queries)} sub-queries using OpenAI")
                return sub_queries
            else:
                logging.warning(f"Invalid format returned from OpenAI: {result}")
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON from OpenAI response: {result}")
            
        # If JSON parsing fails, try to extract queries using string processing
        import re
        sub_queries = re.findall(r'"([^"]*)"', result)
        if sub_queries:
            logging.info(f"Extracted {len(sub_queries)} sub-queries using regex")
            return sub_queries
            
        # If all else fails, return the original query
        logging.warning("Could not extract sub-queries, using original query")
        return [query]
    except Exception as e:
        logging.error(f"Error in LLM planning: {e}")
        # Fall back to placeholder function
        return llm_placeholder_plan(query, current_findings)

def llm_synthesize(query: str, all_results: Dict[str, List[Dict[str, str]]], all_content: Dict[str, Optional[str]]) -> str:
    """
    Real implementation of LLM-based synthesis using Azure OpenAI.
    Takes all gathered info and generates a structured report.
    """
    client = init_azure_openai_client()
    if not client:
        logging.warning("Using placeholder synthesis function as OpenAI client initialization failed")
        return llm_placeholder_synthesize(query, all_results, all_content)
    
    logging.info("Using real OpenAI model for synthesis")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    try:
        # Create a system message for synthesis
        system_message = """You are a research synthesis assistant. 
Your task is to create a comprehensive, well-structured research report based on search results and content. 
Include relevant information, organize it logically, and cite sources. 
Format the report in Markdown with appropriate sections and subsections."""
        
        # Prepare search results summary
        results_summary = ""
        total_results = 0
        for engine_name, results in all_results.items():
            results_summary += f"Results from {engine_name} ({len(results)} results):\n"
            for i, res in enumerate(results[:5]):  # Limit to first 5 results per engine to save tokens
                total_results += 1
                title = res.get('title', 'N/A')
                link = res.get('link', res.get('href', 'N/A'))
                snippet = res.get('snippet', res.get('body', 'N/A'))
                results_summary += f"- Title: {title}\n  Link: {link}\n  Snippet: {snippet[:200]}...\n"
            if len(results) > 5:
                results_summary += f"  ... and {len(results) - 5} more results\n"
        
        # Prepare content summary from fetched webpages
        content_summary = ""
        if all_content:
            content_summary = "Content summaries from top sources:\n"
            for i, (url, content) in enumerate(list(all_content.items())[:3]):  # Limit to first 3 content items
                if content:
                    content_summary += f"Source {i+1} ({url}):\n{content[:500]}...\n\n"
        
        user_message = f"""Main research query: "{query}"

Search Results Summary (Total: {total_results}):
{results_summary}

{content_summary}

Please synthesize this information into a comprehensive research report about "{query}".
Structure the report with clear sections, include key findings, and cite sources where appropriate.
Format the report in Markdown."""
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=10000
        )
        
        report = response.choices[0].message.content.strip()
        logging.info(f"Generated synthesis report ({len(report)} chars) using OpenAI")
        return report
    except Exception as e:
        logging.error(f"Error in LLM synthesis: {e}")
        # Fall back to placeholder function
        return llm_placeholder_synthesize(query, all_results, all_content)

def llm_reflect(current_findings: str) -> Tuple[bool, List[str]]:
    """
    Real implementation of LLM-based reflection using Azure OpenAI.
    Analyzes findings and decides if more research is needed.
    Returns: (needs_more_research: bool, new_queries: List[str])
    """
    client = init_azure_openai_client()
    if not client:
        logging.warning("Using placeholder reflection function as OpenAI client initialization failed")
        return llm_placeholder_reflect(current_findings)
    
    logging.info("Using real OpenAI model for reflection")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    try:
        # Create a system message for reflection
        system_message = """You are a research reflection assistant.
Your task is to analyze the current research findings and determine if more research is needed.
If more research is needed, provide additional queries that could fill gaps in the current findings.
Return your analysis as a JSON object with two fields:
1. "needs_more_research": a boolean indicating if more research is needed (true or false)
2. "new_queries": an array of strings containing new research queries (empty if no more research needed)"""
        
        # Truncate findings if too long
        findings_summary = current_findings[:3000] if len(current_findings) > 3000 else current_findings
        
        user_message = f"""Current research findings:
{findings_summary}

Please analyze these findings and determine:
1. Is the information comprehensive, or are there significant gaps that require more research?
2. If more research is needed, what specific queries would help fill these gaps?

Return ONLY a JSON object with two fields:
- "needs_more_research": boolean
- "new_queries": array of strings
"""
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the JSON result
        try:
            import json
            reflection_result = json.loads(result)
            needs_more = reflection_result.get("needs_more_research", True)
            new_queries = reflection_result.get("new_queries", [])
            
            logging.info(f"Reflection result: needs_more={needs_more}, new_queries={len(new_queries)}")
            return needs_more, new_queries
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON from OpenAI reflection response: {result}")
            # Fall back to default behavior
            return True, []
    except Exception as e:
        logging.error(f"Error in LLM reflection: {e}")
        # Fall back to placeholder function
        return llm_placeholder_reflect(current_findings)

# --- Search Engines ---
class SearchEngine(ABC):
    """Search engine abstract base class"""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """Performs search and returns a list of results."""
        pass

    def __str__(self) -> str:
        return f"SearchEngine({self.name})"

class ArxivSearch(SearchEngine):
    """arXiv Search Engine Implementation"""
    def __init__(self):
        super().__init__("arXiv")
        self.client = arxiv.Client()

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """Uses the arxiv library to search for papers."""
        results = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            api_results = list(self.client.results(search))

            for result in api_results:
                results.append({
                    "title": result.title,
                    "link": result.entry_id,
                    "snippet": result.summary.replace('\n', ' '),
                    "published": str(result.published),
                    "authors": ", ".join([author.name for author in result.authors])
                })
        except Exception as e:
            print(f"[!] Error searching {self.name} for '{query}': {e}")
        return results

class WikipediaSearch(SearchEngine):
    """Wikipedia Search Engine Implementation (using 'wikipedia' library)"""
    def __init__(self, lang: str = 'en'):
        """Initializes Wikipedia search."""
        super().__init__("Wikipedia")
        self.lang = lang
        wikipedia.set_lang(self.lang)
        print(f"[*] Wikipedia language set to: {self.lang}")

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """Searches Wikipedia using the 'wikipedia' library."""
        results = []
        try:
            # wikipedia.search returns a list of potential page titles
            search_titles = wikipedia.search(query, results=num_results)
            if not search_titles:
                print(f"[*] No Wikipedia page titles found for '{query}'")
                return []

            for title in search_titles:
                try:
                    # Attempt to get the page object for each title
                    page = wikipedia.page(title, auto_suggest=False, redirect=True) # Handle redirects
                    results.append({
                        "title": page.title,
                        "link": page.url,
                        "snippet": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                    })
                except wikipedia.exceptions.PageError:
                    print(f"[!] Wikipedia PageError for title '{title}' (query: '{query}'). Skipping.")
                except wikipedia.exceptions.DisambiguationError as e:
                    print(f"[!] Wikipedia DisambiguationError for title '{title}' (query: '{query}'). Options: {e.options[:3]}... Skipping.")
                    # Could potentially try searching one of the options e.options[0]
                except Exception as e_page:
                     print(f"[!] Error fetching Wikipedia page '{title}' (query: '{query}'): {e_page}")

                if len(results) >= num_results:
                     break # Stop if we have enough results

        except Exception as e:
            print(f"[!] Error searching {self.name} for '{query}': {e}")
        return results

class GoogleSearch(SearchEngine):
    """Google Search Engine Implementation using API-based search"""
    def __init__(self):
        super().__init__("Google")
        print("[*] Using Google search via API rather than web scraping")

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """Performs search using the google_search_api function"""
        try:
            api_results = google_search_api(query, max_results=num_results, lang="en")
            # Results are already in the needed format for DeepResearchAgent
            return api_results
        except Exception as e:
            print(f"[!] Error using Google search API for '{query}': {e}")
            return []

class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo Search Engine Implementation using the DDGS API"""
    def __init__(self):
        super().__init__("DuckDuckGo")

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
        """Performs search using the DuckDuckGo API via DDGS library."""
        try:
            api_results = duckduckgo_search_api(query, max_results=num_results)
            formatted_results = []
            
            for result in api_results:
                formatted_results.append({
                    "title": result.get('title', 'No title'),
                    "link": result.get('href', ''),
                    "snippet": result.get('body', 'No description')
                })
            
            return formatted_results
        except Exception as e:
            print(f"[!] Error using DuckDuckGo API for '{query}': {e}")
            return []

# --- Content Fetcher ---

class ContentFetcher:
    """Fetches and extracts main text content from a URL."""
    def fetch_content(self, url: str) -> Optional[str]:
        """Gets URL content and extracts text using the robust API-based implementation."""
        try:
            # Call our implementation from the search API
            content = fetch_webpage_content_api(url, max_length=5000)
            return content
        except Exception as e:
            print(f"[!] Error fetching content from {url}: {e}")
            return None

# --- Updated Research Planner with real LLM integration ---
class ResearchPlanner: 
    def plan(self, main_query: str, current_findings: str) -> List[str]:
        """Uses LLM to generate sub-queries."""
        return llm_plan(main_query, current_findings)


class Reflection:
    """Evaluates research progress using LLM placeholder."""
    def reflect(self, current_findings: str) -> Tuple[bool, List[str]]:
        """Uses LLM placeholder to decide if more research is needed."""
        return llm_reflect(current_findings)


class Synthesizer:
    """Synthesizes the final report using LLM placeholder."""
    def synthesize(self, query: str, all_results: Dict[str, List[Dict[str, str]]], all_content: Dict[str, Optional[str]]) -> str:
        """Uses LLM placeholder to generate the report."""
        # Combine all text for potential input to reflection/synthesis LLM
        findings_text = json.dumps(all_results, indent=2) + "\n\n" + json.dumps(all_content, indent=2)
        # In a real scenario, you might pass 'findings_text' or summaries to the LLM
        return llm_synthesize(query, all_results, all_content)


# --- Main Agent Class (Iterative) ---

class DeepResearchAgent:
    """Orchestrates the iterative research process."""
    def __init__(self, engines: List[SearchEngine], max_iterations: int = MAX_ITERATIONS, fetch_top_n_content: int = 1):
        """Initializes the agent."""
        self.planner = ResearchPlanner()
        self.search_engines = engines
        self.content_fetcher = ContentFetcher()
        self.reflector = Reflection()
        self.synthesizer = Synthesizer()
        self.max_iterations = max_iterations
        self.fetch_top_n_content = fetch_top_n_content

        print(f"[*] DeepResearchAgent initialized with engines: {[e.name for e in engines]}")
        print(f"[*] Max iterations: {self.max_iterations}, Fetch content per engine: {self.fetch_top_n_content}")

    def research(self, query: str, num_results_per_engine: int = DEFAULT_NUM_RESULTS) -> str:
        """Performs iterative research and returns a report."""
        print(f"\n=== Starting Iterative Research for: '{query}' ===")

        # Research State
        initial_query = query
        queries_to_search: Set[str] = {initial_query}
        searched_queries: Set[str] = set()
        all_search_results: Dict[str, List[Dict[str, str]]] = {engine.name: [] for engine in self.search_engines}
        all_fetched_content: Dict[str, Optional[str]] = {}
        urls_fetched: Set[str] = set()
        total_results_count = 0

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # --- Planning ---
            # Combine current findings into a string for context (simplified)
            current_findings_summary = f"Iteration: {iteration+1}. Total Results: {total_results_count}."
            # In a real system, this would be a summary of text content or key points
            if iteration == 0: # Only plan based on initial query in first iteration for this placeholder
                 planned_queries = self.planner.plan(initial_query, "")
            else:
                 # Subsequent iterations might refine or use reflection results
                 # For placeholder, we just use queries generated by reflection (if any) or stop
                 planned_queries = list(queries_to_search - searched_queries) # Use remaining queries

            # Filter out already searched queries
            current_batch_queries = [q for q in planned_queries if q not in searched_queries]
            if not current_batch_queries:
                 print("[*] No new queries to search in this iteration.")
                 # Check reflection to see if we should stop anyway
                 findings_for_reflection = json.dumps(all_search_results) # Simplified findings
                 needs_more, _ = self.reflector.reflect(findings_for_reflection)
                 if not needs_more:
                      print("[*] Reflection suggests stopping.")
                      break
                 else:
                      print("[*] Reflection suggests continuing, but no new queries planned. Stopping.")
                      break # Stop if reflection wants more but planner gave nothing new

            print(f"[*] Queries for this iteration: {current_batch_queries}")
            queries_to_search.update(current_batch_queries) # Add planned to the set

            # --- Searching ---
            new_urls_to_fetch = set()
            for sub_query in current_batch_queries:
                if sub_query in searched_queries: continue # Should not happen with above filter, but safety check
                print(f"  - Searching for: '{sub_query}'")
                searched_queries.add(sub_query)

                for engine in self.search_engines:
                    if total_results_count >= MAX_TOTAL_RESULTS:
                        print(f"[!] Reached max total results ({MAX_TOTAL_RESULTS}). Skipping further searches.")
                        break # Stop searching if max total results reached

                    print(f"    - Using {engine.name}...")
                    results = engine.search(sub_query, num_results=num_results_per_engine)
                    print(f"    - Found {len(results)} results.")
                    if results:
                        # Append results, ensuring not to exceed MAX_TOTAL_RESULTS
                        results_to_add = results[:max(0, MAX_TOTAL_RESULTS - total_results_count)]
                        all_search_results[engine.name].extend(results_to_add)
                        added_count = len(results_to_add)
                        total_results_count += added_count

                        # Collect URLs to fetch from newly added results
                        if self.fetch_top_n_content > 0:
                            count = 0
                            for res in results_to_add:
                                if count >= self.fetch_top_n_content: break
                                link = res.get('link')
                                if link and link.startswith('http') and not link.lower().endswith('.pdf') and link not in urls_fetched:
                                    new_urls_to_fetch.add(link)
                                    count += 1
                if total_results_count >= MAX_TOTAL_RESULTS: break # Break outer loop too

            # --- Content Fetching ---
            if new_urls_to_fetch:
                print(f"[*] Fetching content from {len(new_urls_to_fetch)} new URLs...")
                for url in new_urls_to_fetch:
                    if url in urls_fetched: continue # Should not happen with above check
                    print(f"  - Fetching: {url}")
                    content = self.content_fetcher.fetch_content(url)
                    all_fetched_content[url] = content
                    urls_fetched.add(url)
                    if content: print(f"  - Fetched content (length: {len(content)}).")
                    else: print(f"  - Failed to fetch content.")
            else:
                print("[*] No new URLs to fetch content from in this iteration.")

            # --- Reflection ---
            # Prepare findings summary for reflection (simplified)
            findings_for_reflection = json.dumps(all_search_results) # Could also include fetched content summaries
            needs_more_research, new_queries_from_reflection = self.reflector.reflect(findings_for_reflection)

            if new_queries_from_reflection:
                 new_unsearched = [q for q in new_queries_from_reflection if q not in searched_queries and q not in queries_to_search]
                 if new_unsearched:
                      print(f"[*] Reflection generated new queries: {new_unsearched}")
                      queries_to_search.update(new_unsearched) # Add new queries for next iteration planning

            if not needs_more_research:
                 print("[*] Reflection suggests stopping research after this iteration.")
                 break # Exit loop based on reflection

            if iteration == self.max_iterations - 1:
                 print("[*] Reached maximum iterations.")
                 break # Exit loop if max iterations reached

        # --- Synthesis ---
        print("\n--- Synthesizing Final Report ---")
        final_report = self.synthesizer.synthesize(initial_query, all_search_results, all_fetched_content)

        print(f"\n=== Research Finished for: '{initial_query}' ===")
        return final_report

# --- Main Execution Block ---
if __name__ == "__main__":
    # IMPORTANT: Install the 'wikipedia' library: pip install wikipedia
    search_engines_to_use = [
        ArxivSearch(),
        WikipediaSearch(lang='en'), # Using the new implementation
        DuckDuckGoSearch(),
        GoogleSearch() # Google is now using API-based implementation
    ]

    # Create Agent instance
    agent = DeepResearchAgent(
        engines=search_engines_to_use,
        max_iterations=2, # Limit iterations for testing
        fetch_top_n_content=1 # Fetch content for top 1 result per engine/query
    )
    research_topic = "in LLM, what is decoder-only model and what is attention?"

    report = agent.research(research_topic, num_results_per_engine=5) # Limit results per query

    print("\n" + "="*50)
    print("Generated Research Report:")
    print("="*50)
    print(report)

    try:
        filename = f"research_report_v2_{research_topic.replace(' ', '_').lower()}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[*] Report saved to '{filename}'")
    except Exception as e:
        print(f"[!] Error saving report to file: {e}")


