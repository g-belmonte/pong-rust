//! 3D mesh loaders (OBJ + glTF), both gated behind Cargo features.
//!
//! Two-tier API:
//!
//! - **Low-level:** [`parse_obj`] / [`parse_gltf`] return `Vec<MeshData>` —
//!   typed per-attribute arrays the caller can inspect, transform, or repack
//!   into a custom vertex layout.
//! - **High-level:** [`pack_lit_vertices`] takes a `MeshData` and produces a
//!   [`ModelMesh`] in the engine's lit vertex layout (`pos: vec3 + normal:
//!   vec3 + uv: vec2`, stride 32). [`Resources::load_obj`] /
//!   [`Resources::load_gltf`](crate::resources::Resources::load_gltf) chain
//!   both together to produce RAII [`Mesh`](crate::resources::Mesh) handles
//!   ready to draw against the lit material from test-3d (or any material
//!   declaring the same `vertex_attrs`).
//!
//! What the loaders fill in when source data is missing:
//!
//! - **Normals:** smoothed area-weighted vertex normals (sum face normals,
//!   normalise). For files that bake hard creases as duplicate vertices, the
//!   smoothing is a no-op on the duplicates; for files that share vertices
//!   across smooth surfaces, this gives the natural smoothed result. To get
//!   *flat* normals from a single-index source, the caller has to split
//!   vertices before calling [`pack_lit_vertices`].
//! - **UVs:** `(0, 0)` for every vertex. The lit material doesn't sample a
//!   texture today, so the UVs go unused; a future texture-sampling lit
//!   material can call the low-level path and provide real UVs.
//! - **Indices:** sequential `(0..vertex_count)` if the source has none
//!   (unindexed glTF primitives).
//!
//! ## glTF support
//!
//! The glTF parser handles `.glb` (binary, self-contained) cleanly. `.gltf`
//! JSON files with **embedded base64 data URI buffers** also work; `.gltf`
//! files that reference external `.bin` buffers do **not** — the loader
//! receives a byte slice with no filesystem context to resolve URIs against.
//! GLB is the recommended format for engine consumption; multi-file `.gltf`
//! authoring workflows should export to GLB before shipping.

use crate::graphics_manager::structures::ModelMesh;

/// Per-attribute mesh data, as returned by the low-level parsers. All arrays
/// are vertex-aligned (same length); `indices` references into them.
#[derive(Clone, Debug)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

/// Pack a [`MeshData`] into the engine's lit vertex layout:
/// `pos: vec3 + normal: vec3 + uv: vec2`, stride 32 bytes, indices `u32`.
/// Matches `crates/engine/shaders/src/lit.vert` and the test-3d cube material.
///
/// Panics if `positions`, `normals`, and `uvs` aren't the same length — the
/// loaders always produce parallel arrays, but a caller building a `MeshData`
/// by hand could violate that invariant.
pub fn pack_lit_vertices(data: &MeshData) -> ModelMesh {
    assert_eq!(
        data.positions.len(),
        data.normals.len(),
        "pack_lit_vertices: positions and normals must be the same length"
    );
    assert_eq!(
        data.positions.len(),
        data.uvs.len(),
        "pack_lit_vertices: positions and uvs must be the same length"
    );

    let vertex_count = data.positions.len();
    let stride = 32;
    let mut vertex_bytes = Vec::with_capacity(vertex_count * stride);
    for i in 0..vertex_count {
        vertex_bytes.extend_from_slice(unsafe {
            ::std::slice::from_raw_parts(data.positions[i].as_ptr() as *const u8, 12)
        });
        vertex_bytes.extend_from_slice(unsafe {
            ::std::slice::from_raw_parts(data.normals[i].as_ptr() as *const u8, 12)
        });
        vertex_bytes.extend_from_slice(unsafe {
            ::std::slice::from_raw_parts(data.uvs[i].as_ptr() as *const u8, 8)
        });
    }
    ModelMesh {
        vertex_bytes,
        vertex_stride: stride as u32,
        indices: data.indices.clone(),
    }
}

/// Smoothed area-weighted vertex normals: sum each triangle's geometric
/// normal into its three corners, then normalise per vertex. Triangles with
/// degenerate area contribute zero.
#[cfg(any(feature = "obj", feature = "gltf"))]
fn generate_smoothed_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0_f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0.max(i1).max(i2) >= positions.len() {
            continue;
        }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        // Face normal = e1 × e2 (not yet unit-length; magnitude = 2 × area,
        // which gives the area weighting for free).
        let face = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &idx in &[i0, i1, i2] {
            normals[idx][0] += face[0];
            normals[idx][1] += face[1];
            normals[idx][2] += face[2];
        }
    }
    for n in normals.iter_mut() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }
    normals
}

#[cfg(feature = "obj")]
pub fn parse_obj(bytes: &[u8]) -> Vec<MeshData> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(bytes);
    let (models, _materials) = tobj::load_obj_buf(
        &mut cursor,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
        // No material loading — game code that wants OBJ materials should use
        // tobj directly. The engine's lit material doesn't sample anything.
        |_| Err(tobj::LoadError::OpenFileFailed),
    )
    .expect("parse_obj: failed to parse OBJ bytes");

    let mut out = Vec::with_capacity(models.len());
    for model in models {
        let mesh = model.mesh;
        let positions: Vec<[f32; 3]> = mesh
            .positions
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let uvs: Vec<[f32; 2]> = if mesh.texcoords.is_empty() {
            vec![[0.0, 0.0]; positions.len()]
        } else {
            mesh.texcoords
                .chunks_exact(2)
                .map(|c| [c[0], c[1]])
                .collect()
        };
        let normals: Vec<[f32; 3]> = if mesh.normals.is_empty() {
            generate_smoothed_normals(&positions, &mesh.indices)
        } else {
            mesh.normals
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect()
        };
        out.push(MeshData {
            positions,
            normals,
            uvs,
            indices: mesh.indices,
        });
    }
    out
}

#[cfg(feature = "gltf")]
pub fn parse_gltf(bytes: &[u8]) -> Vec<MeshData> {
    let gltf = gltf::Gltf::from_slice(bytes).expect("parse_gltf: failed to parse glTF bytes");
    // `Gltf::from_slice` populates `.blob` only for GLB. .gltf JSON with
    // *embedded* (data: URI) buffers will route through `Buffer::Source::Uri`
    // below; .gltf with *external* .bin files has no filesystem context here
    // and panics with a clear message.
    let blob = gltf.blob.as_deref();

    // Resolve each declared buffer to a byte slice ahead of time so the
    // per-primitive reader callback is a HashMap-style lookup.
    let mut buffer_data: Vec<Vec<u8>> = Vec::with_capacity(gltf.buffers().len());
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                let b = blob.expect(
                    "parse_gltf: glTF references the binary chunk but the blob is missing (was this parsed from .gltf without a buffer?)",
                );
                buffer_data.push(b.to_vec());
            }
            gltf::buffer::Source::Uri(uri) => {
                panic!(
                    "parse_gltf: external glTF buffer URI {uri:?} not supported — convert the asset to .glb or embed buffers as data: URIs"
                );
            }
        }
    }

    let mut out = Vec::new();
    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| buffer_data.get(buf.index()).map(|v| v.as_slice()));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .expect("parse_gltf: primitive without POSITION attribute")
                .collect();

            let indices: Vec<u32> = if let Some(idx) = reader.read_indices() {
                idx.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };

            let uvs: Vec<[f32; 2]> = if let Some(t) = reader.read_tex_coords(0) {
                t.into_f32().collect()
            } else {
                vec![[0.0, 0.0]; positions.len()]
            };

            let normals: Vec<[f32; 3]> = if let Some(n) = reader.read_normals() {
                n.collect()
            } else {
                generate_smoothed_normals(&positions, &indices)
            };

            out.push(MeshData {
                positions,
                normals,
                uvs,
                indices,
            });
        }
    }
    out
}
