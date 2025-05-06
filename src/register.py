# src/register.py
"""
Scans given content source folders (hard-coded) plus the local `content/` directory,
loads all JSON definitions into Pydantic models, and registers them in a central registry.
Ignores any subfolder named "meta" or starting with a dot.
Supports special handling for models with an `id` field: those are stored in a dict by id, overriding earlier definitions when duplicates occur; other models are stored in lists.
"""
import sys
import json
from pathlib import Path
import inspect
from typing import List, Dict, Union

from pydantic import BaseModel

# Hard-coded content source directories (relative to project root)
MOD_PATHS = [
    Path(__file__).resolve().parent.parent / "content_custom" / "modA",
    Path(__file__).resolve().parent.parent / "content_custom" / "modB",
]
# Local content directory
LOCAL_CONTENT = Path(__file__).resolve().parent.parent / "content"

REGISTRY: Dict[str, Union[List[BaseModel], Dict[str, BaseModel]]] = {}

def is_valid_folder(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.name.startswith('.')
        and path.name != 'meta'
    )


def load_models() -> Dict[str, BaseModel.__class__]:
    """
    Dynamically import all Pydantic model classes from src/objects.py.
    Returns a mapping of model_name -> class.
    """
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_path))
    try:
        objs_mod = __import__("objects", fromlist=["*"])
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import objects module: {e}")

    models: Dict[str, BaseModel.__class__] = {}
    for name, cls in inspect.getmembers(objs_mod, inspect.isclass):
        if issubclass(cls, BaseModel) and cls is not BaseModel:
            models[name] = cls
    return models


def register_content(folders: List[Path]) -> Dict[str, Union[List[BaseModel], Dict[str, BaseModel]]]:
    """
    Load all JSON files in each valid subfolder of the given folders,
    parse them with the corresponding Pydantic model based on folder name,
    and collect them into a registry dict:
      - For models with an `id` field: { model_name: { id: instance, ... } }
      - For others: { model_name: [instance, ...] }
    """
    models = load_models()
    # Determine which models use `id` as a key
    id_models = {name for name, cls in models.items() if 'id' in getattr(cls, '__fields__', {})}

    # Initialize registry with appropriate structures
    registry: Dict[str, Union[List[BaseModel], Dict[str, BaseModel]]] = {}
    for name in models:
        if name in id_models:
            registry[name] = {}
        else:
            registry[name] = []

    # Load in order: mods first, then local content
    for folder in folders:
        if not folder.exists():
            continue
        for sub in folder.iterdir():
            if not is_valid_folder(sub):
                continue
            model_name = sub.name
            model_cls = models.get(model_name)
            if model_cls is None:
                # skip unknown model folders
                continue
            # Scan JSON files in this model's folder
            for json_file in sorted(sub.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    instance = model_cls.model_validate(data)
                    if model_name in id_models:
                        key = getattr(instance, 'id')
                        registry[model_name][key] = instance
                    else:
                        registry[model_name].append(instance)
                except Exception as e:
                    print(f"Error parsing {json_file}: {e}")

    return registry


def main():
    global REGISTRY
    # Combine mod paths and local content
    all_sources = [LOCAL_CONTENT] + MOD_PATHS
    REGISTRY = register_content(all_sources)
    # Summary output
    for model_name, collection in REGISTRY.items():
        count = len(collection)
        kind = 'entries'
        print(f"Loaded {count} {model_name} {kind}.")


if __name__ == "__main__":
    main()
