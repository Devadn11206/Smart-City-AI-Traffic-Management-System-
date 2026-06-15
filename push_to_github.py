#!/usr/bin/env python3
"""
Simple GitHub Push Script
Pushes Smart City AI project to GitHub repository
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and report status"""
    print(f"\n{'='*60}")
    if description:
        print(f"📌 {description}")
    print(f"$ {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, cwd=project_dir)
    return result.returncode == 0

# Project directory
project_dir = Path('c:/Users/ddnan/intenrship_IIIT')

print("\n🚀 SMART CITY AI - GITHUB PUSH")
print(f"Project: {project_dir}")
print(f"Repository: https://github.com/Devadn11206/Smart-City-AI-Traffic-Management-System-.git")

# Step 1: Initialize git (if needed)
print("\n[1/5] Checking git initialization...")
if not (project_dir / '.git').exists():
    run_command("git init", "Initializing git repository")
else:
    print("✓ Git already initialized")

# Step 2: Configure remote
print("\n[2/5] Configuring remote...")
run_command('git remote remove origin 2>nul || true', "Removing old remote (if exists)")
run_command(
    'git remote add origin https://github.com/Devadn11206/Smart-City-AI-Traffic-Management-System-.git',
    "Adding GitHub repository as origin"
)

# Step 3: Stage all files
print("\n[3/5] Staging all files...")
run_command("git add .", "Adding all files to staging area")

# Step 4: Create commit
print("\n[4/5] Creating commit...")
commit_msg = """Smart City AI: Hybrid RL+PINN+ACO Emergency Response System

- Reinforcement Learning agents (DQN, DDQN, PPO)
- Physics-Informed Neural Networks (PINN) for congestion prediction
- Ant Colony Optimization (ACO) for dynamic routing
- Integrated Emergency Corridor with traffic clearing
- 33% improvement in emergency response time
- Complete visualization, validation, and ablation study framework
- Production-ready system with 1000+ step stability verification"""

run_command(f'git commit -m "{commit_msg}"', "Creating commit")

# Step 5: Push to GitHub
print("\n[5/5] Pushing to GitHub...")
run_command("git branch -M main", "Setting main branch")
result = run_command("git push -u origin main", "Pushing to GitHub")

# Final status
print("\n" + "="*60)
if result:
    print("✅ SUCCESS: Project pushed to GitHub!")
    print(f"📍 Repository: https://github.com/Devadn11206/Smart-City-AI-Traffic-Management-System-")
else:
    print("⚠️  Check your authentication:")
    print("   - Personal access token (if using HTTPS)")
    print("   - SSH key (if using SSH)")
    print("\nTo create a personal access token:")
    print("   1. Go to GitHub Settings → Developer settings → Personal access tokens")
    print("   2. Generate new token with 'repo' scope")
    print("   3. Use token as password when prompted")

print("="*60)
