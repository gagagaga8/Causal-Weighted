"""
PubMedliterature searchservice
 useNCBI E-utilities API Rowpaper 
"""

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

PUBMED_API_KEY = os.getenv('PUBMED_API_KEY', '')
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedService:
    """PubMed literature search service"""
    
    def __init__(self):
        self.api_key = PUBMED_API_KEY
        
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        searchPubMed 
        
        Args:
            query: search 
            max_results: ReturnMaximumResults 
            
        Returns:
             ColumnTable
        """
        # 1. search PMIDColumnTable
        search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
            
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            
            # 2. 
            return self._fetch_details(pmids)
            
        except Exception as e:
            print(f"PubMedsearchError: {e}")
            return []
    
    def _fetch_details(self, pmids: List[str]) -> List[Dict]:
        """ Info"""
        fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        if self.api_key:
            fetch_params["api_key"] = self.api_key
            
        try:
            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            response.raise_for_status()
            
            # parseXML
            return self._parse_xml(response.text)
            
        except Exception as e:
            print(f" Error: {e}")
            return []
    
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """parsePubMed XMLresponse"""
        import xml.etree.ElementTree as ET
        
        papers = []
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                paper = {}
                
                # PMID
                pmid_elem = article.find(".//PMID")
                paper["pmid"] = pmid_elem.text if pmid_elem is not None else ""
                
                # title
                title_elem = article.find(".//ArticleTitle")
                paper["title"] = title_elem.text if title_elem is not None else ""
                
                # Authors
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)
                paper["authors"] = authors[:5] # only 5 Authors
                
                journal_elem = article.find(".//Journal/Title")
                paper["journal"] = journal_elem.text if journal_elem is not None else ""
                
                year_elem = article.find(".//PubDate/Year")
                paper["year"] = year_elem.text if year_elem is not None else ""
                
                # Summary
                abstract_elem = article.find(".//Abstract/AbstractText")
                paper["abstract"] = abstract_elem.text if abstract_elem is not None else ""
                
                # DOI
                doi_elem = article.find(".//ArticleId[@IdType='doi']")
                paper["doi"] = doi_elem.text if doi_elem is not None else ""
                
                # PubMed 
                paper["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                
                papers.append(paper)
                
        except Exception as e:
            print(f"XMLparseError: {e}")
            
        return papers
    
    def search_rrt_related(self, max_results: int = 20) -> List[Dict]:
        """searchRRTCorrelation """
        query = "(renal replacement therapy[Title/Abstract]) AND (acute kidney injury[Title/Abstract]) AND (machine learning OR reinforcement learning OR decision support)"
        return self.search(query, max_results)
    
    def search_by_keywords(self, keywords: List[str], max_results: int = 10) -> List[Dict]:
        """according to ColumnTablesearch"""
        query = " AND ".join([f"({kw}[Title/Abstract])" for kw in keywords])
        return self.search(query, max_results)


pubmed_service = PubMedService()


def get_pubmed_service() -> PubMedService:
    return pubmed_service
