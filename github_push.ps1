#!/usr/bin/env powershell
# =============================================================================
# github_push.ps1
# One-time GitHub setup and push script for SIH26051 project
#
# BEFORE RUNNING:
#   1. Install Git: winget install --id Git.Git -e  (needs admin PowerShell)
#   2. Create an empty repo on https://github.com  (no README, no .gitignore)
#   3. Set GITHUB_USERNAME and REPO_NAME below
#   4. Run this script from the project root in a new terminal after Git install
# =============================================================================

param(
    [string]$RepoUrl        = "",
    [string]$GitHubUsername = "",
    [string]$RepoName       = "compliance-risk-clusters",
    [string]$CommitMsg      = "feat: complete implementation for SIH26051 compliance risk clustering"
)

# Ensure MinGit is in PATH if not already
$gitCmdDir = "$env:LOCALAPPDATA\Programs\MinGit\cmd"
if (Test-Path $gitCmdDir) {
    $env:Path = "$gitCmdDir;$env:Path"
}

# Resolve RepoUrl
if (-not $RepoUrl) {
    if ($GitHubUsername) {
        $RepoUrl = "https://github.com/$GitHubUsername/$RepoName.git"
    } else {
        $RepoUrl = Read-Host "Enter your GitHub Repository URL (e.g. https://github.com/username/repo.git)"
    }
}

if (-not $RepoUrl) {
    Write-Error "No repository URL provided. Exiting."
    exit 1
}

Write-Host "`n=== SIH26051 GitHub Push Script ===" -ForegroundColor Cyan
Write-Host "Repository : $RepoUrl"
Write-Host "Commit msg : $CommitMsg`n"

# --- Configure Git identity (skip if already set) ---------------------------
$gitName  = git config --global user.name  2>$null
$gitEmail = git config --global user.email 2>$null

if (-not $gitName) {
    $gitName = Read-Host "Enter your Git user name"
    git config --global user.name  $gitName
}
if (-not $gitEmail) {
    $gitEmail = Read-Host "Enter your Git email"
    git config --global user.email $gitEmail
}

# --- Initialise repo (idempotent) -------------------------------------------
if (-not (Test-Path ".git")) {
    git init
    Write-Host "Initialised new Git repository." -ForegroundColor Green
} else {
    Write-Host "Git repository already initialised." -ForegroundColor Yellow
}

# --- Stage and commit -------------------------------------------------------
git add -A
$status = git status --short
if (-not $status) {
    Write-Host "Nothing to commit — working tree is clean." -ForegroundColor Yellow
} else {
    git commit -m $CommitMsg
    Write-Host "Committed changes." -ForegroundColor Green
}

# --- Set remote and push ----------------------------------------------------
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "Remote 'origin' already set to: $existingRemote"
    $change = Read-Host "Change to $RepoUrl? (y/n)"
    if ($change -eq "y") {
        git remote set-url origin $RepoUrl
    }
} else {
    git remote add origin $RepoUrl
    Write-Host "Remote 'origin' set to $RepoUrl" -ForegroundColor Green
}

git branch -M main
Write-Host "`nPushing to GitHub..." -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Push successful! View your repo at:" -ForegroundColor Green
    Write-Host "   https://github.com/$GitHubUsername/$RepoName" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Push failed. Common fixes:" -ForegroundColor Red
    Write-Host "   - Make sure the GitHub repo exists and is EMPTY"
    Write-Host "   - Use a Personal Access Token (PAT) as your password"
    Write-Host "   - GitHub PAT docs: https://docs.github.com/en/authentication"
}
