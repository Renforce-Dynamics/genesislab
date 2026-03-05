#!/bin/bash
set -e

# ============================================================
# Config
# ============================================================

EXT_YAML_FILE="${EXT_SETUP_YAML:-$(dirname "$0")/setup_ext.yaml}"

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
        echo "📦 No editable Python modules configured."
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

# Parse setup_ext.yaml and emit:
#   REPO lines:   REPO<TAB>url<TAB>path<TAB>editable
#   MODULE lines: MODULE<TAB>path
parse_ext_yaml() {
    python - << 'PY'
import os

yaml_path = os.environ.get("EXT_YAML_FILE")
if not yaml_path or not os.path.isfile(yaml_path):
    raise SystemExit(f"YAML file not found: {yaml_path!r}")

repos = []
editable_modules = []

current_section = None
current_repo = None

with open(yaml_path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        stripped = line.lstrip()

        # Section headers
        if stripped.startswith("repos:"):
            if current_repo:
                repos.append(current_repo)
                current_repo = None
            current_section = "repos"
            continue
        if stripped.startswith("editable_modules:"):
            if current_repo:
                repos.append(current_repo)
                current_repo = None
            current_section = "editable_modules"
            continue

        # Inside repos list
        if current_section == "repos":
            if stripped.startswith("- "):
                if current_repo:
                    repos.append(current_repo)
                current_repo = {}
                # allow "- name: xxx" on same line
                parts = stripped[2:].split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"').strip("'")
                    current_repo[key] = value
                continue

            if ":" in stripped and not stripped.startswith("repos:"):
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                current_repo = current_repo or {}
                current_repo[key] = value
                continue

        # Inside editable_modules list
        if current_section == "editable_modules" and stripped.startswith("- "):
            path = stripped[2:].strip().strip('"').strip("'")
            if path:
                editable_modules.append(path)

    if current_repo:
        repos.append(current_repo)

for repo in repos:
    url = repo.get("url", "")
    path = repo.get("path", "")
    editable_raw = str(repo.get("editable", "")).lower()
    editable = "true" if editable_raw in ("true", "1", "yes", "y") else "false"
    if url and path:
        print(f"REPO\t{url}\t{path}\t{editable}")

for mod in editable_modules:
    print(f"MODULE\t{mod}")
PY
}

# ============================================================
# Main script
# ============================================================

echo "🧩 Setting up required external components (this step is mandatory)..."
echo "🧩 Reading config from $EXT_YAML_FILE ..."

if [ ! -f "$EXT_YAML_FILE" ]; then
    echo "❌ Required config file not found: $EXT_YAML_FILE"
    exit 1
fi

editable_modules=()

echo "🧩 Checking and cloning required repositories..."

while IFS=$'\t' read -r kind field2 field3 field4; do
    [ -z "$kind" ] && continue

    if [ "$kind" = "REPO" ]; then
        repo_url="$field2"
        repo_path="$field3"
        editable="$field4"

        clone_repo "$repo_url" "$repo_path"

        if [ "$editable" = "true" ]; then
            editable_modules+=("$repo_path")
        fi
    elif [ "$kind" = "MODULE" ]; then
        editable_modules+=("$field2")
    fi
done < <(EXT_YAML_FILE="$EXT_YAML_FILE" parse_ext_yaml)

install_modules "${editable_modules[@]}"

echo "✅ External setup completed."
