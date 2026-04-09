#!/usr/bin/env python3
"""Extract static terrain (Floor, Wall, Ceiling) from Scene.usd into a terrain-only file."""

from pxr import Usd, UsdGeom, Sdf
from pathlib import Path

def extract_terrain(input_usd: str, output_usd: str):
    """Extract only the static terrain elements from a USD scene.

    Args:
        input_usd: Path to input Scene.usd
        output_usd: Path to output terrain-only USD file
    """
    # Open source stage
    src_stage = Usd.Stage.Open(input_usd)

    # Create destination stage
    dst_stage = Usd.Stage.CreateNew(output_usd)

    # Copy over the Environment (lights, physics scene)
    env_prim = src_stage.GetPrimAtPath('/Environment')
    if env_prim:
        UsdGeom.Xform.Define(dst_stage, '/Environment')
        dst_stage.GetPrimAtPath('/Environment').GetReferences().AddReference(
            input_usd, '/Environment'
        )

    # Create root
    root = UsdGeom.Xform.Define(dst_stage, '/root')

    # Static terrain elements to extract (no joints)
    terrain_elements = ['Floor', 'Wall', 'Ceiling', 'Table252']

    for elem_name in terrain_elements:
        src_path = f'/root/{elem_name}'
        src_prim = src_stage.GetPrimAtPath(src_path)

        if not src_prim:
            print(f"Warning: {elem_name} not found in source USD")
            continue

        print(f"Copying {elem_name}...")

        # Copy the entire subtree for this element
        dst_path = f'/root/{elem_name}'
        dst_prim = dst_stage.DefinePrim(dst_path)
        dst_prim.GetReferences().AddReference(input_usd, src_path)

    # Set default prim
    dst_stage.SetDefaultPrim(root.GetPrim())

    # Save
    dst_stage.Save()
    print(f"\n✅ Terrain-only USD saved to: {output_usd}")
    print(f"   Contains: {', '.join(terrain_elements)}")


if __name__ == "__main__":
    input_file = "third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Scene.usd"
    output_file = "third_party/genPiHub/data/assets/CWDL_LW_Assets_20260310/Terrain.usd"

    extract_terrain(input_file, output_file)
