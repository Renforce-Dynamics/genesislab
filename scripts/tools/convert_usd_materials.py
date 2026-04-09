#!/usr/bin/env python3
"""Convert Omniverse MDL materials to UsdPreviewSurface.

USD files created in Isaac Sim / Omniverse use OmniPBR.mdl materials,
which require omniverse-kit to bake. Genesis shows them as white without it.

This script reads MDL material inputs (diffuse_texture, etc.) and creates
equivalent UsdPreviewSurface shaders so Genesis can display textures correctly.

Usage:
    python scripts/utils/convert_usd_materials.py --input Scene.usd --output Scene_preview.usd
    python scripts/utils/convert_usd_materials.py --input Terrain.usd --output Terrain_preview.usd
"""

import argparse
import os
from pathlib import Path
from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf, Ar


def get_mdl_input(shader_prim, input_name):
    """Get an input value from an MDL shader prim."""
    attr = shader_prim.GetAttribute(f"inputs:{input_name}")
    if attr and attr.IsValid():
        return attr.Get()
    return None


def resolve_asset_path(asset_path, stage) -> str:
    """Resolve an Sdf.AssetPath to an absolute file path.

    Isaac Sim USD files store texture paths as relative paths inside the USD.
    After exporting to a new location, those relative paths break.
    This function returns the absolute path so the exported USD always works.

    Priority:
    1. asset_path.resolvedPath  -- populated by USD resolver when stage is open
    2. Manually join with layer directory  -- fallback for unresolved relative paths
    3. Raw path string  -- last resort
    """
    if asset_path is None:
        return None

    # resolvedPath is the absolute path the USD resolver found
    resolved = asset_path.resolvedPath
    if resolved:
        return resolved

    # Fallback: resolve relative to the root layer's directory
    raw = asset_path.path
    if not raw:
        return None

    layer_dir = os.path.dirname(stage.GetRootLayer().identifier)
    candidate = os.path.normpath(os.path.join(layer_dir, raw))
    if os.path.exists(candidate):
        return candidate

    # Last resort: return raw path as-is
    return raw


def create_preview_surface_for_material(stage, material_prim, shader_prim):
    """Replace MDL shader with UsdPreviewSurface on the same material."""
    mat_path = material_prim.GetPath()

    # Read MDL inputs before clearing
    diffuse_texture_raw = get_mdl_input(shader_prim, "diffuse_texture")
    diffuse_texture = resolve_asset_path(diffuse_texture_raw, stage)
    diffuse_color = get_mdl_input(shader_prim, "diffuse_tint") or (0.8, 0.8, 0.8)
    roughness = get_mdl_input(shader_prim, "reflection_roughness_constant")
    if roughness is None:
        roughness = 0.5
    metallic = get_mdl_input(shader_prim, "metallic_constant")
    if metallic is None:
        metallic = 0.0
    normal_texture_raw = get_mdl_input(shader_prim, "normalmap_texture")
    normal_texture = resolve_asset_path(normal_texture_raw, stage)

    # Remove old MDL shader
    stage.RemovePrim(shader_prim.GetPath())

    # Create new UsdPreviewSurface shader at same path
    ps_path = mat_path.AppendChild("PreviewSurface")
    ps_prim = UsdShade.Shader.Define(stage, ps_path)
    ps_prim.CreateIdAttr("UsdPreviewSurface")

    # Set roughness and metallic
    ps_prim.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    ps_prim.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

    if diffuse_texture:
        # Create a texture reader for diffuse
        tex_path = mat_path.AppendChild("DiffuseTexture")
        tex_prim = UsdShade.Shader.Define(stage, tex_path)
        tex_prim.CreateIdAttr("UsdUVTexture")
        tex_prim.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture)
        tex_prim.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_prim.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

        # Create UV reader
        uv_path = mat_path.AppendChild("UVReader")
        uv_prim = UsdShade.Shader.Define(stage, uv_path)
        uv_prim.CreateIdAttr("UsdPrimvarReader_float2")
        uv_prim.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex_prim.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_prim.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )

        # Connect texture to surface diffuse
        rgb_out = tex_prim.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        ps_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    else:
        # Use solid color
        if hasattr(diffuse_color, '__iter__'):
            color = tuple(float(c) for c in list(diffuse_color)[:3])
        else:
            color = (0.8, 0.8, 0.8)
        ps_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))

    if normal_texture:
        # Create normal map reader
        nm_path = mat_path.AppendChild("NormalTexture")
        nm_prim = UsdShade.Shader.Define(stage, nm_path)
        nm_prim.CreateIdAttr("UsdUVTexture")
        nm_prim.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_texture)
        nm_prim.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        nm_prim.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        nm_prim.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

        uv_path2 = mat_path.AppendChild("UVReaderNormal")
        uv_prim2 = UsdShade.Shader.Define(stage, uv_path2)
        uv_prim2.CreateIdAttr("UsdPrimvarReader_float2")
        uv_prim2.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        nm_prim.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_prim2.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        rgb_out_n = nm_prim.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        ps_prim.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(rgb_out_n)

    # Connect PreviewSurface output to material surface
    ps_out = ps_prim.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat = UsdShade.Material(material_prim)
    mat.CreateSurfaceOutput().ConnectToSource(ps_out)

    return diffuse_texture is not None


def convert_materials(input_path: str, output_path: str) -> None:
    """Convert all MDL materials in a USD file to UsdPreviewSurface."""
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    # Open stage
    stage = Usd.Stage.Open(input_path)

    # Find all materials
    all_materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    print(f"\nFound {len(all_materials)} materials")

    converted = 0
    skipped = 0

    for mat_prim in all_materials:
        mat = UsdShade.Material(mat_prim)

        # Check if this is an MDL material (sourceAsset)
        shader_prims = [
            p for p in Usd.PrimRange(mat_prim)
            if p.IsA(UsdShade.Shader)
        ]

        mdl_shaders = []
        for sp in shader_prims:
            shader = UsdShade.Shader(sp)
            if shader.GetImplementationSource() == "sourceAsset":
                mdl_shaders.append(sp)

        if not mdl_shaders:
            # Already UsdPreviewSurface or no shader
            skipped += 1
            continue

        print(f"\n  Converting: {mat_prim.GetPath()}")
        for mdl_shader in mdl_shaders:
            has_texture = create_preview_surface_for_material(stage, mat_prim, mdl_shader)
            if has_texture:
                print(f"    → UsdPreviewSurface with diffuse texture")
            else:
                print(f"    → UsdPreviewSurface with solid color")
        converted += 1

    print(f"\nConverted: {converted} materials")
    print(f"Skipped:   {skipped} materials (already standard)")

    # Save to output
    stage.Export(output_path)
    print(f"\n✅ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Omniverse MDL to UsdPreviewSurface")
    parser.add_argument("--input", required=True, help="Input USD file path")
    parser.add_argument("--output", required=True, help="Output USD file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    convert_materials(str(input_path), str(args.output))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
