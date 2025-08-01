#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Demo
Demonstrates the value discovery and execution system.
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_discovery_demo():
    """Run a simplified value discovery demonstration."""
    print("🔍 Terragon Autonomous SDLC Demo")
    print("=" * 50)
    
    # Simulate value discovery
    print("1. 🔎 Discovering work items from multiple sources...")
    
    discovered_items = [
        {
            "id": "demo-001",
            "title": "Update cryptography dependency",
            "category": "security",
            "priority_score": 85.2,
            "estimated_effort": 2.0,
            "source": "dependency_scan"
        },
        {
            "id": "demo-002", 
            "title": "Add type hints to scanner module",
            "category": "code_quality",
            "priority_score": 67.5,
            "estimated_effort": 4.0,
            "source": "static_analysis"
        },
        {
            "id": "demo-003",
            "title": "Refactor authentication module",
            "category": "technical_debt", 
            "priority_score": 72.1,
            "estimated_effort": 6.0,
            "source": "git_history"
        },
        {
            "id": "demo-004",
            "title": "Add performance benchmarks",
            "category": "performance",
            "priority_score": 58.3,
            "estimated_effort": 3.0,
            "source": "performance_monitoring"
        }
    ]
    
    print(f"   ✅ Discovered {len(discovered_items)} work items")
    
    # Show prioritization
    print("\n2. 🎯 Prioritizing by composite value score...")
    sorted_items = sorted(discovered_items, key=lambda x: x['priority_score'], reverse=True)
    
    print("   Top priority items:")
    for i, item in enumerate(sorted_items[:3], 1):
        print(f"   {i}. [{item['id']}] {item['title']}")
        print(f"      Score: {item['priority_score']:.1f} | Category: {item['category'].title()}")
        print(f"      Effort: {item['estimated_effort']}h | Source: {item['source'].replace('_', ' ').title()}")
    
    # Simulate execution
    print(f"\n3. 🚀 Executing highest-value item: {sorted_items[0]['title']}")
    
    # Create a simple execution demonstration
    demo_execution = {
        "work_item_id": sorted_items[0]['id'],
        "title": sorted_items[0]['title'],
        "start_time": datetime.now().isoformat(),
        "changes_made": [
            "Created security advisory documentation",
            "Updated dependency tracking spreadsheet", 
            "Added vulnerability assessment notes"
        ],
        "validation_passed": True,
        "branch_created": f"auto-value/{sorted_items[0]['id']}-update-cryptography",
        "execution_time": 2.3,
        "success": True
    }
    
    print(f"   📝 Created branch: {demo_execution['branch_created']}")
    print(f"   🔧 Made {len(demo_execution['changes_made'])} changes:")
    for change in demo_execution['changes_made']:
        print(f"      - {change}")
    print(f"   ✅ Validation passed: {demo_execution['validation_passed']}")
    print(f"   ⏱️ Execution time: {demo_execution['execution_time']}s")
    
    # Show metrics
    print("\n4. 📊 Value Metrics Summary")
    print(f"   • Items Discovered: {len(discovered_items)}")
    print(f"   • Security Items: {len([i for i in discovered_items if i['category'] == 'security'])}")
    print(f"   • Technical Debt Items: {len([i for i in discovered_items if i['category'] == 'technical_debt'])}")
    print(f"   • Average Priority Score: {sum(i['priority_score'] for i in discovered_items)/len(discovered_items):.1f}")
    print(f"   • High-Value Items (>70): {len([i for i in discovered_items if i['priority_score'] > 70])}")
    
    # Create demo backlog
    print("\n5. 📄 Generated BACKLOG.md")
    create_demo_backlog(discovered_items, demo_execution)
    
    print("\n🎉 Demo completed successfully!")
    print("\n📚 Next Steps:")
    print("   • Review generated BACKLOG.md")
    print("   • Check `.terragon/` configuration files")
    print("   • Run `python3 .terragon/value-discovery.py` for real discovery")
    print("   • Set up autonomous scheduling with `python3 .terragon/scheduler.py`")


def create_demo_backlog(items, execution):
    """Create a demonstration backlog file."""
    backlog_content = f"""# 📊 Terragon Autonomous SDLC Demo Backlog

Generated: {datetime.now().isoformat()}
Status: **DEMONSTRATION MODE**

## 🎯 Recently Executed
**[{execution['work_item_id'].upper()}] {execution['title']}**
- ✅ **Status**: Completed Successfully
- **Execution Time**: {execution['execution_time']}s
- **Changes Made**: {len(execution['changes_made'])} files
- **Branch**: `{execution['branch_created']}`

## 📋 Discovered Work Items

| Rank | ID | Title | Score | Category | Est. Hours | Source |
|------|-----|--------|---------|----------|------------|---------|
"""
    
    sorted_items = sorted(items, key=lambda x: x['priority_score'], reverse=True)
    for i, item in enumerate(sorted_items, 1):
        title_short = item['title'][:50] + "..." if len(item['title']) > 50 else item['title']
        backlog_content += f"| {i} | {item['id']} | {title_short} | {item['priority_score']:.1f} | {item['category'].replace('_', ' ').title()} | {item['estimated_effort']:.1f} | {item['source'].replace('_', ' ').title()} |\n"
    
    backlog_content += f"""

## 📈 Discovery Statistics
- **Total Items**: {len(items)}
- **Security Items**: {len([i for i in items if i['category'] == 'security'])}
- **Technical Debt**: {len([i for i in items if i['category'] == 'technical_debt'])}
- **Code Quality**: {len([i for i in items if i['category'] == 'code_quality'])}
- **Performance**: {len([i for i in items if i['category'] == 'performance'])}
- **Average Score**: {sum(i['priority_score'] for i in items)/len(items):.1f}

## 🔄 Discovery Sources
- **Security Scans**: {len([i for i in items if i['source'] == 'security_scan'])} items
- **Static Analysis**: {len([i for i in items if i['source'] == 'static_analysis'])} items  
- **Git History**: {len([i for i in items if i['source'] == 'git_history'])} items
- **Performance Monitoring**: {len([i for i in items if i['source'] == 'performance_monitoring'])} items
- **Dependency Scans**: {len([i for i in items if i['source'] == 'dependency_scan'])} items

## ⚙️ System Configuration
- **Repository Maturity**: Maturing (50-75%)
- **Scoring Model**: WSJF + ICE + Technical Debt
- **Execution Mode**: Autonomous with validation
- **Min Score Threshold**: 15.0
- **Security Boost**: 2.5x

## 🤖 Autonomous Capabilities

### ✅ Currently Operational
- Multi-source value discovery
- Intelligent prioritization (WSJF + ICE + Technical Debt)
- Automated execution with validation
- Pull request creation and review assignment
- Continuous learning and adaptation
- Safety mechanisms and rollback

### 📅 Execution Schedule
- **Immediate**: On PR merge events
- **Hourly**: Security vulnerability scans  
- **Daily 02:00**: Comprehensive analysis
- **Weekly Mon 03:00**: Deep SDLC assessment
- **Monthly 1st 04:00**: Strategic review

---
**Demo Status**: This is a demonstration of the Terragon Autonomous SDLC system.  
**Real Mode**: Run `python3 .terragon/value-discovery.py` for actual discovery.  
**Autonomous Mode**: Run `python3 .terragon/scheduler.py` for continuous operation.

*Generated by Terragon Autonomous SDLC v1.0.0*
"""
    
    Path("BACKLOG.md").write_text(backlog_content)
    print("   ✅ Created demonstration BACKLOG.md")


if __name__ == "__main__":
    run_discovery_demo()