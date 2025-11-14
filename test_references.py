#!/usr/bin/env python3
"""Test script to verify the references tab functionality"""

import requests
import json

def test_references_endpoint():
    """Test the corpus-documents endpoint that feeds the references tab"""
    try:
        response = requests.get("http://localhost:8000/corpus-documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])

            print("✅ References endpoint working!")
            print(f"Found {len(documents)} documents")

            for doc in documents:
                title = doc.get('title', 'Unknown')
                text_length = doc.get('text_length', 'MISSING')
                source = doc.get('source', 'Unknown')
                year = doc.get('year', '')

                if text_length == 'MISSING':
                    print(f"❌ ERROR: 'text_length' field missing for document '{title}'")
                    return False
                else:
                    print(f"✅ Document: {title}")
                    print(f"   Source: {source} | Year: {year} | Text Length: {text_length} characters")

            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing references functionality...")
    success = test_references_endpoint()
    if success:
        print("\n🎉 References tab should now work without the 'text_length' error!")
    else:
        print("\n💥 References tab still has issues.")
