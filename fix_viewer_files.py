#!/usr/bin/env python3
"""Quick fix: patch existing HTML files to embed JSON data instead of fetching it"""

import json
from pathlib import Path

viewer_dir = Path("analysis_results/unified_viewer_lite")

for html_file in viewer_dir.rglob("image_*.html"):
    json_file = html_file.parent / f"{html_file.stem}_data.json"
    
    if not json_file.exists():
        print(f"✗ {html_file.name} - no JSON file found")
        continue
    
    # Load HTML and JSON
    with open(html_file, 'r') as f:
        html_content = f.read()
    
    # Skip if already fixed
    if 'const allData = {' in html_content and 'async function loadData()' not in html_content:
        print(f"✓ {html_file.name} already fixed")
        continue
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Find the script section and replace it
    script_start = html_content.find('<script>')
    script_end = html_content.find('</script>')
    
    if script_start == -1 or script_end == -1:
        print(f"✗ {html_file.name} - couldn't find script tags")
        continue
    
    # Get the parts
    before_script = html_content[:script_start + 8]  # Include <script>
    after_script = html_content[script_end:]
    
    # Get old script to extract some values
    old_script = html_content[script_start + 8:script_end]
    
    # Extract gridSize, availableLayers, and currentLayer from old script
    import re
    grid_match = re.search(r'let gridSize = (\d+);', old_script)
    grid_size = grid_match.group(1) if grid_match else str(data['grid_size'])
    
    layers_match = re.search(r'const availableLayers = (\{[^;]+\});', old_script)
    avail_layers = layers_match.group(1) if layers_match else json.dumps(data['available_layers'])
    
    layer_match = re.search(r'let currentLayer = (\d+);', old_script)
    current_layer = layer_match.group(1) if layer_match else '0'
    
    # Create new script content with embedded data
    new_script = f'''
        // Embed all data directly in HTML
        const allData = {json.dumps(data['unified_patch_data'])};
        const interpretabilityData = {json.dumps(data['interpretability_map'])};
        const gridSize = {grid_size};
        const availableLayers = {avail_layers};
        
        let currentLayer = {current_layer};
        let activePatchDiv = null;
        let activePatchIdx = null;
        let patches = {{}};
        
''' + old_script[old_script.find('// Create patch overlays'):]
    
    # Remove loadData function and fix initialization
    new_script = new_script.replace(
        '// Initialize - load data first\n        loadData();',
        '''// Initialize
        const baseImage = document.getElementById('baseImage');
        if (baseImage.tagName === 'IMG') {
            baseImage.addEventListener('load', createPatches);
            if (baseImage.complete) { createPatches(); }
        } else {
            createPatches();
        }'''
    )
    
    # Remove null check in createPatches
    new_script = new_script.replace(
        '''if (!allData) {
                console.log('Data not loaded yet');
                return;
            }
            
            ''',
        ''
    )
    
    # Fix resize listener
    new_script = new_script.replace(
        '''if (allData) {  // Only recreate patches if data is loaded
                setTimeout(createPatches, 100);
            }''',
        'setTimeout(createPatches, 100);'
    )
    
    # Reconstruct HTML
    new_html = before_script + new_script + after_script
    
    # Write back
    with open(html_file, 'w') as f:
        f.write(new_html)
    
    print(f"✓ Fixed {html_file.name}")

print("\n✅ All files patched!")
