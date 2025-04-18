# src/content_env.py
"""
Generates JSON Schema files from all Pydantic BaseModel classes
found in src/objects.py, placing each schema under content/meta/<ClassName>/schema.json
"""
import sys
import importlib
import inspect
import json
from pathlib import Path
from pydantic import BaseModel


def main():
    # Determine project structure paths
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    # Import the module where your Pydantic models live
    module_name = "objects"
    try:
        objs = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Error: Could not import module '{module_name}' from {src_path}")
        sys.exit(1)

    # Base output directory for schemas
    output_base = project_root / "content" / "meta"
    output_base.mkdir(parents=True, exist_ok=True)

    # Iterate through all classes in objects.py
    for name, cls in inspect.getmembers(objs, inspect.isclass):
        # Select only Pydantic models (exclude BaseModel itself)
        if issubclass(cls, BaseModel) and cls is not BaseModel:
            if name[0] == "_":
                continue
            
            # Generate the JSON Schema dict
            schema_dict = cls.model_json_schema()

            # Prepare output folder: content/meta/<ClassName>/
            model_dir = output_base / name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Write schema.json
            schema_file = model_dir / "schema.json"
            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump(schema_dict, f, indent=2)

            print(f"✔ Wrote schema for '{name}' to {schema_file}")


if __name__ == "__main__":
    main()
