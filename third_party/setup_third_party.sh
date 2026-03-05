#!/bin/bash
set -e

# ============================================================
# Config
# ============================================================

YAML_FILE="${THIRD_PARTY_YAML:-$(dirname "$0")/third_party_repos.yaml"

# ============================================================
# Utility functions
# ============================================================

# Clone a repository if it doesn't already exist
clone_repo() {
    local repo_url=$1
    local target_dir=$2

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ Cloning $repo_url to $target_dir ..."
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
    else
        echo "✔ Repository already exists at $target_dir, skipping clone."
    fi
}

# Check and install local editable Python packages
install_modules() {
    local modules=("$@")

    if [ "${#modules[@]}" -eq 0 ]; then
        echo "📦 No editable Python modules to install (editable: true)."
        return 0
    fi

    echo "📦 Installing local editable modules..."
    for module in "${modules[@]}"; do
        if [ -d "$module" ]; then
            echo "→ Installing $module ..."
            pip install -e "$module"
        else
            echo "⚠ Skipped $module (directory not found)"
        fi
    done
}

# Parse our simple YAML file and emit: URL<TAB>PATH<TAB>EDITABLE
parse_yaml_repos() {
    python - << 'PY'
import os

yaml_path = os.environ.get("YAML_FILE")
if not yaml_path or not os.path.isfile(yaml_path):
    raise SystemExit(f"YAML file not found: {yaml_path!r}")

repos = []
current = None

with open(yaml_path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        stripped = line.lstrip()

        # Start of new item: "- name: xxx" or just "- "
        if stripped.startswith("- "):
            if current:
                repos.append(current)
            current = {}
            # Maybe has a key in same line, e.g. "- name: rsl_rl"
            parts = stripped[2:].split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"').strip("'")
                current[key] = value
            continue

        # Key: value (indented)
        if ":" in stripped and not stripped.startswith("repos:"):
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current = current or {}
            current[key] = value

    if current:
        repos.append(current)

for repo in repos:
    url = repo.get("url", "")
    path = repo.get("path", "")
    editable_raw = str(repo.get("editable", "")).lower()
    editable = "true" if editable_raw in ("true", "1", "yes", "y") else "false"
    if url and path:
        print(f"{url}\t{path}\t{editable}")
PY
}

# ============================================================
# Main script
# ============================================================

echo "🧩 Reading third-party repos from $YAML_FILE ..."

if [ ! -f "$YAML_FILE" ]; then
    echo "❌ YAML config file not found: $YAML_FILE"
    exit 1
fi

echo "🧩 Checking and cloning required repositories..."

editable_modules=()

while IFS=$'\t' read -r repo_url repo_path editable; do
    [ -z "$repo_url" ] && continue

    clone_repo "$repo_url" "$repo_path"

    if [ "$editable" = "true" ]; then
        editable_modules+=("$repo_path")
    fi
done < <(YAML_FILE="$YAML_FILE" parse_yaml_repos)

install_modules "${editable_modules[@]}"

echo "✅ All done!"
