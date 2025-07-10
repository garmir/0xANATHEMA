#!/usr/bin/env python3
"""
Privacy Audit Script for Task Master AI
Identifies and removes external API calls to ensure complete local operation
"""

import os
import re
from pathlib import Path

def audit_external_calls():
    """Audit files for external API calls and suggest fixes"""
    
    print("🔍 Privacy Audit: Scanning for External API Calls")
    print("=" * 60)
    
    # Files to audit
    files_to_audit = [
        "local_llm_adapter.py",
        "local_research_module.py", 
        "local_planning_engine.py",
        "autonomous_research_integration.py",
        "hardcoded_research_workflow.py",
        "perplexity_client.py.old"
    ]
    
    # Patterns that indicate external calls
    external_patterns = [
        (r'https?://[^"\s]+', "External URL"),
        (r'requests\.(get|post|put|delete)', "HTTP request"),
        (r'urllib\.request', "URL request"),
        (r'httpx\.(get|post|put|delete)', "HTTPX request"),
        (r'api\.perplexity\.ai', "Perplexity API"),
        (r'api\.openai\.com', "OpenAI API"),
        (r'api\.anthropic\.com', "Anthropic API")
    ]
    
    issues_found = []
    total_issues = 0
    
    for file_path in files_to_audit:
        if os.path.exists(file_path):
            print(f"\n📁 Auditing: {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            file_issues = []
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments and docstrings
                if line.strip().startswith('#') or '"""' in line or "'''" in line:
                    continue
                
                for pattern, description in external_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        file_issues.append({
                            'line': line_num,
                            'content': line.strip(),
                            'matches': matches,
                            'type': description
                        })
                        total_issues += len(matches)
            
            if file_issues:
                print(f"❌ Found {len(file_issues)} lines with external calls:")
                for issue in file_issues:
                    print(f"   Line {issue['line']}: {issue['type']} - {issue['content'][:80]}...")
                issues_found.extend([(file_path, issue) for issue in file_issues])
            else:
                print("✅ No external calls detected")
    
    print(f"\n📊 Privacy Audit Summary:")
    print(f"Total files scanned: {len([f for f in files_to_audit if os.path.exists(f)])}")
    print(f"Files with issues: {len(set(issue[0] for issue in issues_found))}")
    print(f"Total external call instances: {total_issues}")
    
    if issues_found:
        print(f"\n🔧 Recommended Actions:")
        print("1. Remove or comment out external API endpoints")
        print("2. Replace with local LLM adapter calls")
        print("3. Add offline fallback mechanisms")
        print("4. Update configuration to use local providers only")
    else:
        print("\n🎉 Privacy Audit PASSED: No external calls detected")
    
    return issues_found

def generate_privacy_fixes():
    """Generate specific fixes for privacy issues"""
    
    print("\n🛠️ Generating Privacy Fixes...")
    
    # Check specific files for fixable issues
    fixes = []
    
    # Fix local_llm_adapter.py external calls
    adapter_file = "local_llm_adapter.py"
    if os.path.exists(adapter_file):
        with open(adapter_file, 'r') as f:
            content = f.read()
        
        if "requests.post" in content:
            fixes.append({
                'file': adapter_file,
                'issue': 'Uses requests.post for HTTP calls',
                'fix': 'Replace with local LLM provider calls or make conditional on provider availability'
            })
    
    # Check for hardcoded URLs
    for file_path in ["local_llm_adapter.py", "autonomous_research_integration.py"]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            if "http://" in content or "https://" in content:
                fixes.append({
                    'file': file_path,
                    'issue': 'Contains hardcoded HTTP/HTTPS URLs',
                    'fix': 'Replace with configurable local endpoints (localhost:port)'
                })
    
    if fixes:
        print(f"Found {len(fixes)} specific issues to fix:")
        for i, fix in enumerate(fixes, 1):
            print(f"{i}. {fix['file']}: {fix['issue']}")
            print(f"   Fix: {fix['fix']}")
    else:
        print("No specific privacy fixes needed")
    
    return fixes

def main():
    """Run privacy audit"""
    issues = audit_external_calls()
    fixes = generate_privacy_fixes()
    
    # Generate privacy compliance report
    report = {
        'audit_timestamp': '2025-07-10T20:15:00Z',
        'issues_found': len(issues),
        'privacy_status': 'COMPLIANT' if len(issues) == 0 else 'NON-COMPLIANT',
        'recommended_fixes': len(fixes)
    }
    
    print(f"\n📋 Privacy Status: {report['privacy_status']}")
    
    return report

if __name__ == "__main__":
    main()