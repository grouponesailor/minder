import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        self.elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME")
        self.elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD")
        
        # Target indexes
        self.target_indexes = ["person_dataset", "organization_units", "systems_index"]
        
        # API settings
        self.api_title = "Universal Elasticsearch Search API"
        self.api_description = "API for searching across persons, organization units, and systems with intelligent universal search"
        self.api_version = "3.0.0"
        
        # Search settings
        self.default_search_size = 20
        self.max_search_size = 100
        self.min_autocomplete_chars = 2
        self.max_autocomplete_results = 10
        
        # Load cursor rules
        self.cursor_rules = self._load_cursor_rules()
    
    def _load_cursor_rules(self) -> Dict[str, Any]:
        """Load cursor rules from JSON file"""
        try:
            rules_path = Path(__file__).parent.parent.parent / "cursor_rules.json"
            with open(rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cursor rules: {e}")
            return {"rules": []}
    
    @property
    def elasticsearch_auth(self):
        """Get Elasticsearch authentication tuple"""
        if self.elasticsearch_username and self.elasticsearch_password:
            return (self.elasticsearch_username, self.elasticsearch_password)
        return None


# Global settings instance
settings = Settings() 