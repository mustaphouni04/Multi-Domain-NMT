"""
TMX to CSV Converter
This script converts TMX (Translation Memory eXchange) files to CSV format,
extracting translation pairs along with context information.
Avoids duplications by keeping only unique source-target pairs.
"""

import re
import os
import sys
import csv
from xml.sax.saxutils import unescape

def read_file(file_path):
    """
    Reads a file with the correct encoding (first tries UTF-16, then UTF-8)
    """
    encodings = ['utf-16', 'utf-8', 'latin-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                if content:
                    print(f"File successfully read with encoding {encoding}")
                    break
        except Exception as e:
            print(f"Error reading with encoding {encoding}: {e}")
    
    if not content:
        raise ValueError(f"Could not read the file {file_path} with any of the attempted encodings")
    
    return content

def clean_html_tags(text):
    """
    Cleans any HTML tags from the text.
    
    Args:
        text: Text to clean
        
    Returns:
        Clean text without HTML tags
    """
    if not text:
        return ""
    
    # First decode HTML entities to convert &lt; to < and &gt; to >
    text = unescape(text)
    
    # Remove content between <ph> and </ph> along with the tags
    text = re.sub(r'<ph>.*?</ph>', '', text, flags=re.DOTALL)
    
    # Remove content between <bpt and </bpt> along with the tags
    text = re.sub(r'<bpt[^>]*>.*?</bpt>', '', text, flags=re.DOTALL)
    
    # Remove content between <ept and </ept> along with the tags
    text = re.sub(r'<ept[^>]*>.*?</ept>', '', text, flags=re.DOTALL)
    
    # Remove <it> and </it> tags while keeping their content
    text = re.sub(r'<it[^>]*>(.*?)</it>', r'\1', text, flags=re.DOTALL)
    
    # Remove <seg> and </seg> tags while keeping their content
    text = re.sub(r'<seg>(.*?)</seg>', r'\1', text, flags=re.DOTALL)
    
    return text.strip()

def extract_translation_pairs(xml_content):
    """
    Extracts translation pairs using regex patterns
    """
    translations = []
    counter = 1
    # To track unique source-target pairs
    unique_pairs = set()
    
    # Extract source and target languages
    srclang_match = re.search(r'srclang=["\']([^"\']+)["\']', xml_content)
    targetlang_match = re.search(r'type=["\']targetlang["\']>([^<]+)<', xml_content)
    
    source_language = srclang_match.group(1) if srclang_match else "en"
    target_language = targetlang_match.group(1) if targetlang_match else "es"
    
    print(f"Source language: {source_language}, Target language: {target_language}")
    
    # Find all translation units
    tu_pattern = r'<tu[^>]*>.*?</tu>'
    tus = re.findall(tu_pattern, xml_content, re.DOTALL)
    
    print(f"Found {len(tus)} translation units")
    
    for tu in tus:
        # Extract source segment
        source_pattern = r'<tuv\s+xml:lang=["\']' + source_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        source_match = re.search(source_pattern, tu, re.DOTALL)
        
        # Extract target segment
        target_pattern = r'<tuv\s+xml:lang=["\']' + target_language + r'["\'][^>]*>.*?<seg>(.*?)</seg>'
        target_match = re.search(target_pattern, tu, re.DOTALL)
        
        # Extract context information
        context_pre = ""
        context_post = ""
        
        # Search for context in the source tuv
        source_tuv_pattern = r'<tuv\s+xml:lang=["\']' + source_language + r'["\'][^>]*>(.*?)</tuv>'
        source_tuv_match = re.search(source_tuv_pattern, tu, re.DOTALL)
        
        if source_tuv_match:
            source_tuv_content = source_tuv_match.group(1)
            
            pre_context_pattern = r'<prop\s+type=["\']x-context-pre["\']>(.*?)</prop>'
            post_context_pattern = r'<prop\s+type=["\']x-context-post["\']>(.*?)</prop>'
            
            pre_context_match = re.search(pre_context_pattern, source_tuv_content, re.DOTALL)
            post_context_match = re.search(post_context_pattern, source_tuv_content, re.DOTALL)
            
            if pre_context_match:
                # Clean previous context (remove all HTML tags)
                context_pre = clean_html_tags(pre_context_match.group(1))
            
            if post_context_match:
                # Clean following context (remove all HTML tags)
                context_post = clean_html_tags(post_context_match.group(1))
        
        if source_match and target_match:
            # Extract the original text and clean any HTML tags
            source_text = clean_html_tags(source_match.group(1))
            target_text = clean_html_tags(target_match.group(1))
            
            # Create a unique key for the source-target pair
            pair_key = (source_text, target_text)
            
            # Only add if this pair is not already in our unique pairs set
            if pair_key not in unique_pairs:
                unique_pairs.add(pair_key)
                
                translations.append({
                    'ID': counter,
                    'source': source_text,
                    'source_language': source_language,
                    'target': target_text,
                    'target_language': target_language,
                    'x-context-pre': context_pre,
                    'x-context-post': context_post
                })
                
                counter += 1
    
    print(f"Found {len(unique_pairs)} unique translation pairs out of {len(tus)} total translation units")
    return translations

def save_to_csv(translations, output_file, delimiter=';'):
    """
    Saves translations to a CSV file using the csv library
    which automatically handles escaping issues.
    Uses UTF-8 with BOM encoding for Excel compatibility
    
    Args:
        translations: List of dictionaries with translations
        output_file: Output file path
        delimiter: Delimiter to use (';' is better for Excel in Spanish)
    
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    if not translations:
        print("No translations found!")
        return False
    
    # Define required and optional fields
    required_fields = ['ID', 'source', 'source_language', 'target', 'target_language']
    optional_fields = ['x-context-pre', 'x-context-post']
    all_fields = required_fields + optional_fields
    
    try:
        # Use 'utf-8-sig' instead of 'utf-8' to add the BOM
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # Configure the writer with the appropriate delimiter
            writer = csv.DictWriter(
                csvfile, 
                fieldnames=all_fields,
                delimiter=delimiter,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            
            # Write the header
            writer.writeheader()
            
            # Write the rows
            for translation in translations:
                # Make sure all fields have a value (even if empty)
                row_data = {field: translation.get(field, '') for field in all_fields}
                writer.writerow(row_data)
        
        print(f"Successfully saved {len(translations)} unique translations to {output_file}")
        print(f"The file was saved with UTF-8 with BOM encoding for proper display in Excel")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

input_file = r"../Files/TM/mitsubishi_wsm.tmx"
output_file = r"../Files/TM/mitsubishi_wsm.csv"
delimiter = ';'  # Semicolon for better Excel compatibility

if len(sys.argv) > 1:
    input_file = sys.argv[1]

if len(sys.argv) > 2:
    output_file = sys.argv[2]
    
if len(sys.argv) > 3:
    delimiter = sys.argv[3]

try:
    # Read the TMX file
    xml_content = read_file(input_file)
    
    # Extract translation pairs
    translations = extract_translation_pairs(xml_content)
    
    # Save to CSV
    success = save_to_csv(translations, output_file, delimiter)
    
    if success:
        print(f"Conversion completed successfully. Output saved to {output_file}")
        # Show the first translations as a preview
        for i, trans in enumerate(translations[:3]):
            print(f"\nExample {i+1}:")
            for key, value in trans.items():
                if value:  # Only show fields with content
                    print(f"{key}: {value}")
    else:
        print("Conversion failed.")

except Exception as e:
    print(f"Error processing the TMX file: {e}")
    import traceback
    traceback.print_exc()
